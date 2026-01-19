
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
// 1: ldmatrix with address remap (raw encoding)
// 2: manual lane/group loads (raw encoding)
#define LOAD_METHOD_LDMATRIX_REMAP 1
#define LOAD_METHOD_MANUAL_LANE    2

// ------------------------------------------------------------------------------------------------
// Host Utilities for 2:4 Sparsity
// ------------------------------------------------------------------------------------------------

// Initialize a strictly 2:4 structured sparse matrix A (MxK).
// For each group of 4, exactly two entries are non-zero.
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

// Initialize a dense random matrix
void init_random_matrix(
    std::vector<half>& mat,
    int rows, int cols,
    std::mt19937& gen
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    mat.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = __float2half(dist(gen));
    }
}

// Encode dense matrix A (MxK) to Sparse A (Mx(K/2)) and Metadata E (Mx(K/16))
// Assumes A is already strictly 2:4 structured sparse (two nonzeros per group of 4).
// The metadata is packed: 16 indices (for 16 pairs = 64 original elements) -> 32 bits.
// Actually 1 metadata element (32-bit) covers 32 input elements of A (16 pairs).
void compress_matrix_host(
    const std::vector<half>& A_dense,
    std::vector<half>& A_sparse,
    std::vector<uint32_t>& E_metadata,
    int m, int k
) {
    int k_sparse = k / 2;
    A_sparse.resize(m * k_sparse);

    int meta_cols_packed = k / 32; // Number of uint32_t per row
    E_metadata.assign(m * meta_cols_packed, 0);

    for (int r = 0; r < m; ++r) {
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
                std::cerr << "Invalid 2:4 structure at row " << r
                          << ", group " << c_group << ": nonzeros=" << nz << std::endl;
                exit(EXIT_FAILURE);
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

void cpu_gemm_ref(
    const std::vector<half>& A,
    const std::vector<half>& B,
    std::vector<float>& C,
    int m, int n, int k
) {
    C.assign(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += __half2float(A[i * k + l]) * __half2float(B[l * n + j]);
            }
            C[i * n + j] = sum;
        }
    }
}

int verify_result(const char* tag, const std::vector<float>& gpu, const std::vector<float>& ref, int rows, int cols) {
    int errors = 0;
    for (int i = 0; i < static_cast<int>(gpu.size()); ++i) {
        float diff = std::abs(gpu[i] - ref[i]);
        if (diff > 0.1f) {
            errors++;
            if (errors < 10) {
                std::cout << tag << " mismatch at " << i << " GPU: " << gpu[i]
                          << " CPU: " << ref[i] << std::endl;
            }
        }
    }
    std::cout << tag << " total errors: " << errors << std::endl;
    if (errors > 0) {
        // print_matrix(tag, gpu.data(), rows, cols); // Too big for large matrix
    }
    return errors;
}


// ------------------------------------------------------------------------------------------------
// Device Helpers (Must be defined before kernel)
// ------------------------------------------------------------------------------------------------

// MMA.SP Wrapper
// Using fp16 input, fp32 accumulator.
// Shape: m16n8k32
__device__ __forceinline__ void mma_sp_sync_f32_f16(
    float* d,      // 4 floats (C tile)
    const int* a,  // 4 int32 (16 halfs -> Compressed A) 
    const int* b,  // 4 int32 (16 halfs -> B)
    const float* c,// 4 floats
    const int* e   // 1 int32 (Metadata)
) {
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#endif
}

// Variant without A-reg swap (a0,a1,a2,a3)
__device__ __forceinline__ void mma_sp_sync_f32_f16_a0123(
    float* d,
    const int* a,
    const int* b,
    const float* c,
    const int* e
) {
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3)
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
#endif
}

// Metadata lane mapping for mma.sp.m16n8k32 with sparsity selector f=0
// Threads T0/T1 within each group of 4 lanes provide metadata for rows (groupID, groupID+8).
__device__ __forceinline__ uint32_t load_metadata_for_lane(int lane_id, const uint32_t* smem_E) {
    int groupID = lane_id >> 2;          // 0..7
    int threadID = lane_id & 0x3;        // 0..3
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

// Manual A fragment load for sparse mma.m16n8k32 (raw compressed layout)
__device__ __forceinline__ void load_a_frag_manual(int lane_id, const half* smem_A, int* a_frag) {
    int groupID = lane_id >> 2;         // 0..7
    int threadID = lane_id & 0x3;       // 0..3
    // ai: 0..7 (two values per register)
    half vals[8];
    #pragma unroll
    for (int ai = 0; ai < 8; ++ai) {
        int row = (ai < 2 || (ai >= 4 && ai < 6)) ? groupID : (groupID + 8);
        int col_base = (ai < 4) ? (threadID * 4) : (threadID * 4 + 16);
        int chunk = col_base / 4; // 0..7
        int val_idx = ai & 0x1;   // 0 or 1 (two stored values per 4-wide chunk)
        vals[ai] = smem_A[row * 16 + chunk * 2 + val_idx];
    }
    // Pack two halfs per register (low, high)
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&vals[0]);
    const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&vals[2]);
    const uint32_t* p2 = reinterpret_cast<const uint32_t*>(&vals[4]);
    const uint32_t* p3 = reinterpret_cast<const uint32_t*>(&vals[6]);
    a_frag[0] = p0[0];
    a_frag[1] = p1[0];
    a_frag[2] = p2[0];
    a_frag[3] = p3[0];
}

// Manual B fragment load for mma.m16n8k32 (row-major B, KxN)
__device__ __forceinline__ void load_b_frag_manual(int lane_id, const half* smem_B, int* b_frag) {
    int groupID = lane_id >> 2;         // 0..7
    int threadID = lane_id & 0x3;       // 0..3
    half vals[8];
    #pragma unroll
    for (int bi = 0; bi < 8; ++bi) {
        int row = (threadID * 2) + (bi & 0x1);
        row += (bi / 2) * 8;            // rows 0..31
        int col = groupID;              // N=8 columns
        vals[bi] = smem_B[row * 8 + col]; // 8-column stride default
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
// Device Kernel
// ------------------------------------------------------------------------------------------------

// Single tile M16 N8 K32 MMA demonstrating `mma.sp` loads with metadata.
// This focuses on operator wiring and dataset compression rather than tiling.
template<int LOAD_METHOD>
__global__ void mma_sp_m16n8k32_fp32fp16_kernel(
    const half* __restrict__ A, // Compressed
    const half* __restrict__ B,
    float* __restrict__ C,
    const uint32_t* __restrict__ E,
    int M, int N, int K // Added dims for tiling
) {
    const int M_TILE = 64; 
    const int N_TILE = 64;
    const int K_TILE = 32;
    const int K_TILE_COMPRESSED = 16;
    
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
        // Cooperatively Load Tile to SMEM
        for (int i = tid; i < M_TILE * K_TILE_COMPRESSED; i += blockDim.x) {
            int r = block_row + i / K_TILE_COMPRESSED;
            int c = (k / 2) + i % K_TILE_COMPRESSED;
            smem_A[i] = (r < M && c < K/2) ? A[r*(K/2) + c] : __float2half(0.0f);
        }
        for (int i = tid; i < K_TILE * N_TILE; i += blockDim.x) {
            int r = k + i / N_TILE;
            int c = block_col + i % N_TILE;
            smem_B[i] = (r < K && c < N) ? B[r*N + c] : __float2half(0.0f);
        }
        for (int i = tid; i < M_TILE; i += blockDim.x) {
            int r = block_row + i;
            int c_group = k / 32;
            smem_E[i] = (r < M && c_group < K/32) ? E[r*(K/32) + c_group] : 0;
        }
        __syncthreads();

        // Warp MMA
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 4; ++tile_idx) {
            int cur_warp_col = warp_col_base + tile_idx * 8;
            int a_frag[4]; int b_frag[4]; int e_val;
            
            if (LOAD_METHOD == LOAD_METHOD_MANUAL_LANE) {
                // Determine SMEM pointers for this warp operation
                const half* a_ptr = smem_A + (warp_row_base * K_TILE_COMPRESSED); // 16 rows

                int groupID = lane_id >> 2; int threadID = lane_id & 0x3;
                
                // Load A helper
                load_a_frag_manual(lane_id, a_ptr, a_frag);
                
                // Manual B Load (Inline override for Stride)
                // load_b_frag_manual assumes 8 stride. We have N_TILE=64.
                half vals[8];
                #pragma unroll
                for (int bi = 0; bi < 8; ++bi) {
                    int row = (threadID * 2) + (bi & 0x1) + (bi / 2) * 8;
                    int col = cur_warp_col + groupID; 
                    vals[bi] = smem_B[row * N_TILE + col];
                }
                const uint32_t* p = reinterpret_cast<const uint32_t*>(&vals[0]);
                b_frag[0] = p[0]; b_frag[1] = p[1]; b_frag[2] = p[2]; b_frag[3] = p[3];

                // Load E
                e_val = load_metadata_for_lane(lane_id, smem_E + warp_row_base);
                
                // MMA
                mma_sp_sync_f32_f16_a0123(c_frags[tile_idx], a_frag, b_frag, c_frags[tile_idx], &e_val);

            } else {
                // LDMATRIX not implemented for scaled tiling in this demo
            }
        }
        __syncthreads();
    }

    // Store C
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

// Debug helper
void print_matrix(const char* name, const float* data, int rows, int cols) {
    if (rows > 16 || cols > 16) return;
    std::cout << name << ":" << std::endl;
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            std::cout << std::setw(6) << std::setprecision(2) << data[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 1. Setup Large Problem
    int m = 2048, n = 2048, k = 2048;
    std::cout << "Scaled mma_sp_m16n8k32_fp32fp16 - " << m << "x" << n << "x" << k << std::endl;

    // 2. Alloc
    std::vector<half> h_A_dense(m * k);
    std::vector<half> h_B(k * n);
    
    std::mt19937 gen(42);
    // Use simpler fast init for large arrays
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::cout << "Init..." << std::endl;
    for(size_t i=0; i<h_A_dense.size(); ++i) h_A_dense[i] = __float2half(dist(gen));
    for(size_t i=0; i<h_B.size(); ++i) h_B[i] = __float2half(dist(gen));

    // 3. Compress
    std::cout << "Compress..." << std::endl;
    std::vector<half> h_A_sparse;
    std::vector<uint32_t> h_E;
    // Note: Compress takes time for 2k*2k.
    // Optimization: Create sparse directly? No, preserve 'compress_matrix_host' usage as requested.
    // 'init_structured_sparse_A' is O(MK). OK.
    init_structured_sparse_A(h_A_dense, m, k, gen);
    compress_matrix_host(h_A_dense, h_A_sparse, h_E, m, k);

    // 4. Device Alloc
    half *d_A, *d_B;
    uint32_t *d_E;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, h_A_sparse.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_E, h_E.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A_sparse.data(), h_A_sparse.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, h_E.data(), h_E.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // 5. Run Tiled Kernel
    // Grid: M/64, N/64. Block: 256. SMEM: ~32KB.
    dim3 grid(n/64, m/64);
    dim3 block(256);
    int smem_size = (64*16 + 32*64 + 64) * sizeof(half); // Approx
    if (smem_size < 4000) smem_size = 32768; // Safe margin

    std::cout << "Run GPU..." << std::endl;
    CHECK_CUDA(cudaMemset(d_C, 0, m * n * sizeof(float)));
    mma_sp_m16n8k32_fp32fp16_kernel<LOAD_METHOD_MANUAL_LANE><<<grid, block, smem_size>>>(d_A, d_B, d_C, d_E, m, n, k);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 6. Verify
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
