
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Load methods
#define LOAD_METHOD_LDMATRIX_REMAP 1
#define LOAD_METHOD_MANUAL_LANE    2

// ------------------------------------------------------------------------------------------------
// Host Utilities for 2:4 Sparsity (BF16)
// ------------------------------------------------------------------------------------------------

void init_structured_sparse_A(
    std::vector<__nv_bfloat16>& A_dense,
    int m, int k,
    std::mt19937& gen
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 3);

    A_dense.assign(m * k, __float2bfloat16(0.0f));
    for (int r = 0; r < m; ++r) {
        for (int c_group = 0; c_group < k / 4; ++c_group) {
            int idx0 = pick(gen);
            int idx1 = pick(gen);
            while (idx1 == idx0) idx1 = pick(gen);
            if (idx0 > idx1) std::swap(idx0, idx1);

            float v0 = dist(gen);
            float v1 = dist(gen);
            if (v0 == 0.0f) v0 = 1.0f;
            if (v1 == 0.0f) v1 = -1.0f;

            int base = r * k + c_group * 4;
            A_dense[base + idx0] = __float2bfloat16(v0);
            A_dense[base + idx1] = __float2bfloat16(v1);
        }
    }
}

void compress_matrix_host(
    const std::vector<__nv_bfloat16>& A_dense,
    std::vector<__nv_bfloat16>& A_sparse,
    std::vector<uint32_t>& E_metadata,
    int m, int k
) {
    int k_sparse = k / 2;
    A_sparse.resize(m * k_sparse);

    int meta_cols_packed = k / 32; 
    E_metadata.assign(m * meta_cols_packed, 0);

    for (int r = 0; r < m; ++r) {
        for (int c_group = 0; c_group < k / 4; ++c_group) {
            int base = r * k + c_group * 4;
            int idxs[2] = {-1, -1};
            int nz = 0;
            for (int i = 0; i < 4; ++i) {
                float v = __bfloat162float(A_dense[base + i]);
                if (v != 0.0f) {
                    if (nz < 2) idxs[nz] = i;
                    nz++;
                }
            }
            if (nz != 2) {
                // In random gen, we might get actual 0.0f. Handle gracefully or fail.
                // For this demo, we enforced non-zero.
            }
            if (idxs[0] > idxs[1]) std::swap(idxs[0], idxs[1]);

            int sparse_col_base = c_group * 2;
            A_sparse[r * k_sparse + sparse_col_base + 0] = A_dense[base + idxs[0]];
            A_sparse[r * k_sparse + sparse_col_base + 1] = A_dense[base + idxs[1]];

            int pack_col = c_group / 8;
            int pair_idx = (c_group % 8) * 2;
            uint32_t packed = E_metadata[r * meta_cols_packed + pack_col];
            packed |= (static_cast<uint32_t>(idxs[0]) << (pair_idx * 2));
            packed |= (static_cast<uint32_t>(idxs[1]) << ((pair_idx + 1) * 2));
            E_metadata[r * meta_cols_packed + pack_col] = packed;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Device Helpers
// ------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_sp_sync_f32_bf16(
    float* d,      
    const int* a,  
    const int* b,  
    const float* c,
    const int* e   
) {
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#endif
}

// Variant without A-reg swap (a0,a1,a2,a3) - used for manual load
__device__ __forceinline__ void mma_sp_sync_f32_bf16_a0123(
    float* d,
    const int* a,
    const int* b,
    const float* c,
    const int* e
) {
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#endif
}

__device__ __forceinline__ uint32_t load_metadata_for_lane(int lane_id, const uint32_t* smem_E) {
    int groupID = lane_id >> 2;          
    int threadID = lane_id & 0x3;        
    if (threadID == 0 || threadID == 1) {
        uint32_t e0 = smem_E[groupID];
        uint32_t e1 = smem_E[groupID + 8];
        uint32_t lo0 = e0 & 0xFFFFu;
        uint32_t hi0 = (e0 >> 16) & 0xFFFFu;
        uint32_t lo1 = e1 & 0xFFFFu;
        uint32_t hi1 = (e1 >> 16) & 0xFFFFu;
        if (threadID == 0) return (lo1 << 16) | lo0;
        return (hi1 << 16) | hi0;
    }
    return 0;
}

__device__ __forceinline__ void load_a_frag_manual(int lane_id, const __nv_bfloat16* smem_A, int* a_frag) {
    int groupID = lane_id >> 2;         
    int threadID = lane_id & 0x3;       
    __nv_bfloat16 vals[8];
    #pragma unroll
    for (int ai = 0; ai < 8; ++ai) {
        int row = (ai < 2 || (ai >= 4 && ai < 6)) ? groupID : (groupID + 8);
        int col_base = (ai < 4) ? (threadID * 4) : (threadID * 4 + 16);
        int chunk = col_base / 4; 
        int val_idx = ai & 0x1;   
        vals[ai] = smem_A[row * 16 + chunk * 2 + val_idx];
    }
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&vals[0]);
    const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&vals[2]);
    const uint32_t* p2 = reinterpret_cast<const uint32_t*>(&vals[4]);
    const uint32_t* p3 = reinterpret_cast<const uint32_t*>(&vals[6]);
    a_frag[0] = p0[0];
    a_frag[1] = p1[0];
    a_frag[2] = p2[0];
    a_frag[3] = p3[0];
}

__device__ __forceinline__ void load_b_frag_manual(int lane_id, const __nv_bfloat16* smem_B, int* b_frag) {
    int groupID = lane_id >> 2;         
    int threadID = lane_id & 0x3;       
    __nv_bfloat16 vals[8];
    #pragma unroll
    for (int bi = 0; bi < 8; ++bi) {
        int row = (threadID * 2) + (bi & 0x1);
        row += (bi / 2) * 8;            
        int col = groupID;              
        vals[bi] = smem_B[row * 8 + col];
    }
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&vals[0]);
    const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&vals[2]);
    const uint32_t* p2 = reinterpret_cast<const uint32_t*>(&vals[4]);
    const uint32_t* p3 = reinterpret_cast<const uint32_t*>(&vals[6]);
    b_frag[0] = p0[0];
    b_frag[1] = p1[0];
    b_frag[2] = p2[0];
    b_frag[3] = p3[0];
}

// ------------------------------------------------------------------------------------------------
// Kernel
// ------------------------------------------------------------------------------------------------

template<int LOAD_METHOD>
__global__ void mma_sp_m16n8k32_fp32bf16_kernel(
    const __nv_bfloat16* __restrict__ A, 
    const __nv_bfloat16* __restrict__ B, 
    float* __restrict__ C,
    const uint32_t* __restrict__ E,
    int M, int N, int K 
) {
    const int M_TILE = 64; 
    const int N_TILE = 64;
    const int K_TILE = 32;
    const int K_TILE_COMPRESSED = 16;
    
    int block_row = blockIdx.y * M_TILE;
    int block_col = blockIdx.x * N_TILE;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_A = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_B = (__nv_bfloat16*)(smem + M_TILE * K_TILE_COMPRESSED * sizeof(__nv_bfloat16));
    uint32_t* smem_E = (uint32_t*)(smem + (M_TILE * K_TILE_COMPRESSED + K_TILE * N_TILE) * sizeof(__nv_bfloat16));

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_row_base = (warp_id / 2) * 16; 
    int warp_col_base = (warp_id % 2) * 32;

    float c_frags[4][4]; 
    for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) c_frags[i][j] = 0.0f;

    for (int k = 0; k < K; k += K_TILE) {
        for (int i = tid; i < M_TILE * K_TILE_COMPRESSED; i += blockDim.x) {
            int r = block_row + i / K_TILE_COMPRESSED;
            int c = (k / 2) + i % K_TILE_COMPRESSED;
            smem_A[i] = (r < M && c < K/2) ? A[r*(K/2) + c] : __float2bfloat16(0.0f);
        }
        for (int i = tid; i < K_TILE * N_TILE; i += blockDim.x) {
            int r = k + i / N_TILE;
            int c = block_col + i % N_TILE;
            smem_B[i] = (r < K && c < N) ? B[r*N + c] : __float2bfloat16(0.0f);
        }
        for (int i = tid; i < M_TILE; i += blockDim.x) {
            int r = block_row + i;
            int c_group = k / 32;
            smem_E[i] = (r < M && c_group < K/32) ? E[r*(K/32) + c_group] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int tile_idx = 0; tile_idx < 4; ++tile_idx) {
            int cur_warp_col = warp_col_base + tile_idx * 8;
            int a_frag[4]; int b_frag[4]; int e_val;
            
            if (LOAD_METHOD == LOAD_METHOD_MANUAL_LANE) {
                const __nv_bfloat16* a_ptr = smem_A + (warp_row_base * K_TILE_COMPRESSED); 
                int groupID = lane_id >> 2; int threadID = lane_id & 0x3;
                
                load_a_frag_manual(lane_id, a_ptr, a_frag);
                
                // Manual B Load (Inline override for Stride)
                __nv_bfloat16 vals[8];
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    int row = (threadID * 2) + (bi & 0x1) + (bi / 2) * 8;
                    int col = cur_warp_col + groupID; 
                    vals[bi] = smem_B[row * N_TILE + col];
                }
                const uint32_t* p = reinterpret_cast<const uint32_t*>(&vals[0]);
                b_frag[0] = p[0]; b_frag[1] = p[1]; b_frag[2] = p[2]; b_frag[3] = p[3];

                e_val = load_metadata_for_lane(lane_id, smem_E + warp_row_base);
                
                mma_sp_sync_f32_bf16_a0123(c_frags[tile_idx], a_frag, b_frag, c_frags[tile_idx], &e_val);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tile_idx = 0; tile_idx < 4; ++tile_idx) {
        int cur_warp_col = warp_col_base + tile_idx * 8;
        for (int r = 0; r < 4; ++r) {
            int row = block_row + warp_row_base + (lane_id / 4) + (r >= 2 ? 8 : 0);
            int col = block_col + cur_warp_col + (lane_id % 4) * 2 + (r % 2);
            if (row < M && col < N) C[row * N + col] = c_frags[tile_idx][r];
        }
    }
}

int main() {
    int m = 2048, n = 2048, k = 2048;
    std::cout << "Scaled mma_sp_m16n8k32_fp32bf16 - " << m << "x" << n << "x" << k << std::endl;

    std::vector<__nv_bfloat16> h_A_dense(m * k);
    std::vector<__nv_bfloat16> h_B(k * n);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::cout << "Init..." << std::endl;
    for(size_t i=0; i<h_A_dense.size(); ++i) h_A_dense[i] = __float2bfloat16(dist(gen));
    for(size_t i=0; i<h_B.size(); ++i) h_B[i] = __float2bfloat16(dist(gen));

    std::cout << "Compress..." << std::endl;
    std::vector<__nv_bfloat16> h_A_sparse;
    std::vector<uint32_t> h_E;
    init_structured_sparse_A(h_A_dense, m, k, gen);
    compress_matrix_host(h_A_dense, h_A_sparse, h_E, m, k);

    __nv_bfloat16 *d_A, *d_B;
    uint32_t *d_E;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A_sparse.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_E, h_E.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A_sparse.data(), h_A_sparse.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, h_E.data(), h_E.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 grid(n/64, m/64);
    dim3 block(256);
    int smem_size = 32768; 

    std::cout << "Run GPU..." << std::endl;
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(float)));
    mma_sp_m16n8k32_fp32bf16_kernel<LOAD_METHOD_MANUAL_LANE><<<grid, block, smem_size>>>(d_A, d_B, d_C, d_E, m, n, k);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Verify..." << std::endl;
    std::vector<float> h_C_gpu(m * n);
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Full Verification
    int errs = 0;
    std::cout << "Verifying all " << m * n << " elements..." << std::endl;
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            float ref = 0.0f;
            for (int l = 0; l < k; ++l) {
                ref += __bfloat162float(h_A_dense[r * k + l]) * __bfloat162float(h_B[l * n + c]);
            }
            if (std::abs(h_C_gpu[r * n + c] - ref) > 0.1f) {
                errs++;
                if (errs < 5) std::cout << "Fail at (" << r << "," << c << ") GPU " << h_C_gpu[r * n + c] << " CPU " << ref << std::endl;
            }
        }
        if (r % 256 == 0) std::cout << "Progress: " << (r * 100 / m) << "%" << std::endl;
    }
    std::cout << "Total Errors: " << errs << " / " << m * n << std::endl;

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_E)); CHECK_CUDA(cudaFree(d_C));
    return (errs > 0);
}
