
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
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
#define LOAD_METHOD_LDMATRIX 1
#define LOAD_METHOD_MANUAL   2

// ------------------------------------------------------------------------------------------------
// Host Utilities for 2:4 Sparsity (K=16)
// ------------------------------------------------------------------------------------------------

void init_structured_sparse_A(
    std::vector<half>& A_dense,
    int m, int k,
    std::mt19937& gen
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 3);

    A_dense.assign(m * k, __float2half(0.0f));
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
            A_dense[base + idx0] = __float2half(v0);
            A_dense[base + idx1] = __float2half(v1);
        }
    }
}

void compress_matrix_host(
    const std::vector<half>& A_dense,
    std::vector<half>& A_sparse,
    std::vector<uint32_t>& E_metadata,
    int m, int k
) {
    int k_sparse = k / 2;
    A_sparse.resize(m * k_sparse);

    std::vector<uint16_t> row_meta(m, 0);

    for (int r = 0; r < m; ++r) {
        uint16_t meta = 0;
        for (int c_group = 0; c_group < k / 4; ++c_group) {
            int base = r * k + c_group * 4;
            int idxs[2] = {-1, -1};
            int nz = 0;
            for (int i = 0; i < 4; ++i) {
                float v = __half2float(A_dense[base + i]);
                if (v != 0.0f) {
                    if (nz < 2) idxs[nz] = i;
                    nz++;
                }
            }
            if (nz != 2) {
                // Handle or fail
            }
            if (idxs[0] > idxs[1]) std::swap(idxs[0], idxs[1]);

            int sparse_col_base = c_group * 2;
            A_sparse[r * k_sparse + sparse_col_base + 0] = A_dense[base + idxs[0]];
            A_sparse[r * k_sparse + sparse_col_base + 1] = A_dense[base + idxs[1]];

            int pair_idx = c_group * 2;
            meta |= static_cast<uint16_t>(idxs[0]) << (pair_idx * 2);
            meta |= static_cast<uint16_t>(idxs[1]) << ((pair_idx + 1) * 2);
        }
        row_meta[r] = meta;
    }

    // Pack metadata: 16-bit per row (assuming k=16). 
    // m16n8k16 MMA specifically needs (row i, row i+8) metadata in one uint32_t for a warp of 32 lanes.
    // Each group of 4 lanes processes 2 rows. 
    // For a tile, we store it row-wise but pack it such that warp can load easily.
    // Here we pack row r and r+8 into one uint32_t.
    int num_meta = m * (k/32 > 0 ? k/32 : 1); // For k=16, it's 1 meta per 2 rows.
    E_metadata.assign(m / 2, 0); 
    for (int r = 0; r < m; r += 16) {
        for (int i = 0; i < 8; ++i) {
            uint32_t lo = row_meta[r + i];
            uint32_t hi = row_meta[r + i + 8];
            E_metadata[(r / 2) + i] = (hi << 16) | lo;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Device Helpers
// ------------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t pack_half2(const half lo, const half hi) {
    uint32_t ulo = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo));
    uint32_t uhi = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi));
    return (uhi << 16) | ulo;
}

__device__ __forceinline__ void mma_sp_sync_f32_f16_k16(
    float* d,
    const uint32_t* a,
    const uint32_t* b,
    const float* c,
    const int* e
) {
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11}, %12, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11}, %12, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#endif
}

// ------------------------------------------------------------------------------------------------
// Kernel
// ------------------------------------------------------------------------------------------------

template<int LOAD_METHOD>
__global__ void mma_sp_m16n8k16_fp32fp16_kernel(
    const half* __restrict__ A, 
    const half* __restrict__ B, 
    float* __restrict__ C,
    const uint32_t* __restrict__ E,
    int M, int N, int K 
) {
    const int M_TILE = 64; 
    const int N_TILE = 64;
    const int K_TILE = 16;
    const int K_TILE_COMPRESSED = 8;
    
    int block_row = blockIdx.y * M_TILE;
    int block_col = blockIdx.x * N_TILE;

    extern __shared__ char smem[];
    half* smem_A = (half*)smem;
    half* smem_B = (half*)(smem + M_TILE * K_TILE_COMPRESSED * sizeof(half));
    uint32_t* smem_E = (uint32_t*)(smem + (M_TILE * K_TILE_COMPRESSED + K_TILE * N_TILE) * sizeof(half));

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
            smem_A[i] = (r < M && c < K/2) ? A[r * (K/2) + c] : __float2half(0.0f);
        }
        for (int i = tid; i < K_TILE * N_TILE; i += blockDim.x) {
            int r = k + i / N_TILE;
            int c = block_col + i % N_TILE;
            smem_B[i] = (r < K && c < N) ? B[r * N + c] : __float2half(0.0f);
        }
        for (int i = tid; i < M_TILE / 2; i += blockDim.x) {
            int meta_idx = (block_row / 2) + i;
            // K_TILE=16 means 4 groups of 4. Total 1 uint32_t per 2 rows of 16.
            // Mapping E: M/2 elements per k-strip?
            // Original logic for k=16: E has 8 entries for 16 rows.
            // So for a k-strip, it has M/2 entries.
            int entries_per_k = M / 2;
            int k_strip = k / 16;
            smem_E[i] = (block_row + i < M) ? E[k_strip * entries_per_k + meta_idx] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int tile_idx = 0; tile_idx < 4; ++tile_idx) {
            int cur_warp_col = warp_col_base + tile_idx * 8;
            uint32_t a_frag[2]; uint32_t b_frag[2]; int e_val;
            
            if (LOAD_METHOD == LOAD_METHOD_MANUAL) {
                const half* a_ptr = smem_A + (warp_row_base * K_TILE_COMPRESSED); 
                int group_id = lane_id / 4;       // 0..7
                int col_base = (lane_id % 4) * 2; // 0,2,4,6

                half a0 = a_ptr[group_id * 8 + col_base + 0];
                half a1 = a_ptr[group_id * 8 + col_base + 1];
                half a2 = a_ptr[(group_id + 8) * 8 + col_base + 0];
                half a3 = a_ptr[(group_id + 8) * 8 + col_base + 1];
                a_frag[0] = pack_half2(a0, a1);
                a_frag[1] = pack_half2(a2, a3);

                // Manual B Load with N_TILE stride
                int col_b = cur_warp_col + (lane_id / 4);
                int row_b_base = (lane_id % 4) * 2;
                half b0 = smem_B[row_b_base * N_TILE + col_b];
                half b1 = smem_B[(row_b_base + 1) * N_TILE + col_b];
                half b2 = smem_B[(row_b_base + 8) * N_TILE + col_b];
                half b3 = smem_B[(row_b_base + 9) * N_TILE + col_b];
                b_frag[0] = pack_half2(b0, b1);
                b_frag[1] = pack_half2(b2, b3);

                e_val = smem_E[(warp_row_base / 2) + group_id];
                
                mma_sp_sync_f32_f16_k16(c_frags[tile_idx], a_frag, b_frag, c_frags[tile_idx], &e_val);
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
    std::cout << "Scaled mma_sp_m16n8k16_fp32fp16 - " << m << "x" << n << "x" << k << std::endl;

    std::vector<half> h_A_dense(m * k);
    std::vector<half> h_B(k * n);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::cout << "Init..." << std::endl;
    init_structured_sparse_A(h_A_dense, m, k, gen);
    for(size_t i=0; i<h_B.size(); ++i) h_B[i] = __float2half(dist(gen));

    std::cout << "Compress..." << std::endl;
    std::vector<half> h_A_sparse;
    std::vector<uint32_t> h_E_all;
    
    // For K=2048, there are 128 strips of K=16.
    // Each strip has M/2 metadata entries.
    h_E_all.resize((k / 16) * (m / 2));
    for (int strip = 0; strip < k/16; ++strip) {
        std::vector<half> strip_A_dense(m * 16);
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < 16; ++c) {
                strip_A_dense[r * 16 + c] = h_A_dense[r * k + strip * 16 + c];
            }
        }
        std::vector<half> strip_A_sparse;
        std::vector<uint32_t> strip_E;
        compress_matrix_host(strip_A_dense, strip_A_sparse, strip_E, m, 16);
        
        // Copy sparse part
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < 8; ++c) {
                h_A_sparse.push_back(strip_A_sparse[r * 8 + c]);
            }
        }
        // Copy metadata
        for (int i = 0; i < m/2; ++i) {
            h_E_all[strip * (m/2) + i] = strip_E[i];
        }
    }
    // Sparse A in memory is now [M, K/2] but strip-major? 
    // Re-pack A_sparse to row-major [M, K/2]
    std::vector<half> h_A_sparse_row(m * (k/2));
    for (int strip = 0; strip < k/16; ++strip) {
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < 8; ++c) {
                h_A_sparse_row[r * (k/2) + strip * 8 + c] = h_A_sparse[(strip * m * 8) + (r * 8) + c];
            }
        }
    }

    half *d_A, *d_B;
    uint32_t *d_E;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A_sparse_row.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_E, h_E_all.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A_sparse_row.data(), h_A_sparse_row.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, h_E_all.data(), h_E_all.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 grid(n/64, m/64);
    dim3 block(256);
    int smem_size = (64*8 + 16*64 + 64) * sizeof(half); // Approx
    if (smem_size < 32768) smem_size = 32768;

    std::cout << "Run GPU..." << std::endl;
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(float)));
    mma_sp_m16n8k16_fp32fp16_kernel<LOAD_METHOD_MANUAL><<<grid, block, smem_size>>>(d_A, d_B, d_C, d_E, m, n, k);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Verify..." << std::endl;
    std::vector<float> h_C_gpu(m * n);
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    int errs = 0;
    std::cout << "Verifying all " << m * n << " elements..." << std::endl;
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            float ref = 0.0f;
            for (int l = 0; l < k; ++l) {
                ref += __half2float(h_A_dense[r * k + l]) * __half2float(h_B[l * n + c]);
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
