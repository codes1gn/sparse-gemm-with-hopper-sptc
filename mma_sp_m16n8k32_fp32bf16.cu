
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

// Initialize a dense random matrix
void init_random_matrix(
    std::vector<__nv_bfloat16>& mat,
    int rows, int cols,
    std::mt19937& gen
) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    mat.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = __float2bfloat16(dist(gen));
    }
}

// Encode dense matrix A (MxK) to Sparse A (Mx(K/2)) and Metadata E (Mx(K/16))
// Assumes A is already strictly 2:4 structured sparse (two nonzeros per group of 4).
// The metadata is packed: 16 indices (for 16 pairs = 64 original elements) -> 32 bits.
// Actually 1 metadata element (32-bit) covers 32 input elements of A (16 pairs).
void compress_matrix_host(
    const std::vector<__nv_bfloat16>& A_dense,
    std::vector<__nv_bfloat16>& A_sparse,
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
                float v = __bfloat162float(A_dense[base + i]);
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
    const std::vector<__nv_bfloat16>& A,
    const std::vector<__nv_bfloat16>& B,
    std::vector<float>& C,
    int m, int n, int k
) {
    C.assign(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += __bfloat162float(A[i * k + l]) * __bfloat162float(B[l * n + j]);
            }
            C[i * n + j] = sum;
        }
    }
}

template<int LOAD_METHOD>
__global__ void mma_sp_m16n8k32_fp32bf16_kernel(
    const __nv_bfloat16* __restrict__ A_compressed,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const uint32_t* __restrict__ E
);

void print_matrix(const char* name, const float* data, int rows, int cols);

void run_gpu_demo(
    const std::vector<__nv_bfloat16>& A_sparse,
    const std::vector<__nv_bfloat16>& B,
    const std::vector<uint32_t>& E,
    int m, int n,
    std::vector<float>& out_ldm,
    std::vector<float>& out_manual
) {
    __nv_bfloat16 *d_A, *d_B;
    uint32_t *d_E;
    float *d_C_ldm, *d_C_manual;

    CHECK_CUDA(cudaMalloc(&d_A, A_sparse.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, B.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_E, E.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_C_ldm, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_manual, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A_sparse.data(), A_sparse.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), B.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, E.data(), E.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_C_ldm, 0, m * n * sizeof(float)));
    mma_sp_m16n8k32_fp32bf16_kernel<LOAD_METHOD_LDMATRIX_REMAP><<<1, 32>>>(d_A, d_B, d_C_ldm, d_E);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(d_C_manual, 0, m * n * sizeof(float)));
    mma_sp_m16n8k32_fp32bf16_kernel<LOAD_METHOD_MANUAL_LANE><<<1, 32>>>(d_A, d_B, d_C_manual, d_E);
    CHECK_CUDA(cudaDeviceSynchronize());

    out_ldm.resize(m * n);
    out_manual.resize(m * n);
    CHECK_CUDA(cudaMemcpy(out_ldm.data(), d_C_ldm, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(out_manual.data(), d_C_manual, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_E));
    CHECK_CUDA(cudaFree(d_C_ldm));
    CHECK_CUDA(cudaFree(d_C_manual));
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
        print_matrix(tag, gpu.data(), rows, cols);
    }
    return errors;
}

// ------------------------------------------------------------------------------------------------
// Device Kernel
// ------------------------------------------------------------------------------------------------

// MMA.SP Wrapper
// Using bf16 input, fp32 accumulator.
// Shape: m16n8k32
__device__ __forceinline__ void mma_sp_sync_f32_bf16(
    float* d,      // 4 floats (C tile)
    const int* a,  // 4 int32 (16 halfs -> Compressed A) 
                   // Wait, A is m16k32 -> 16x32 (sparse) -> 16x16 stored? 
                   // Sparse A is M16 x (K32/2) = 16x16.
                   // 16x16 halfs = 256 halfs.
                   // Per thread? No, this is warp-wide operation.
                   // Inputs to mma.sp.sync are usually in registers distributed across warp.
                   // For A (m16k32): 4 registers (32-bit) per thread.
                   // 4 * 32 bits = 128 bits = 8 halfs.
                   // Warp size 32. 32 * 8 = 256 halfs. Matches 16x16.
    const int* b,  // 4 int32 (16 halfs -> B)
                   // B is k32n8 -> 32x8. 
                   // 32*8 = 256 elements.
                   // Warp has 4 registers per thread. Correct.
    const float* c,// 4 floats
    const int* e   // 1 int32 (Metadata)
                   // Metadata covers K=32. Row=16?
                   // E is m16k32 metadata. Size 16 * (32/32) = 16 uint32s?
                   // Distributed: 1 register per thread.
) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
}

// Variant without A-reg swap (a0,a1,a2,a3)
__device__ __forceinline__ void mma_sp_sync_f32_bf16_a0123(
    float* d,
    const int* a,
    const int* b,
    const float* c,
    const int* e
) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
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
__device__ __forceinline__ void load_a_frag_manual(int lane_id, const __nv_bfloat16* smem_A, int* a_frag) {
    int groupID = lane_id >> 2;         // 0..7
    int threadID = lane_id & 0x3;       // 0..3
    // ai: 0..7 (two values per register)
    __nv_bfloat16 vals[8];
    #pragma unroll
    for (int ai = 0; ai < 8; ++ai) {
        int row = (ai < 2 || (ai >= 4 && ai < 6)) ? groupID : (groupID + 8);
        int col_base = (ai < 4) ? (threadID * 4) : (threadID * 4 + 16);
        int chunk = col_base / 4; // 0..7
        int val_idx = ai & 0x1;   // 0 or 1 (two stored values per 4-wide chunk)
        vals[ai] = smem_A[row * 16 + chunk * 2 + val_idx];
    }
    // Pack two bf16 per register (low, high)
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
__device__ __forceinline__ void load_b_frag_manual(int lane_id, const __nv_bfloat16* smem_B, int* b_frag) {
    int groupID = lane_id >> 2;         // 0..7
    int threadID = lane_id & 0x3;       // 0..3
    __nv_bfloat16 vals[8];
    #pragma unroll
    for (int bi = 0; bi < 8; ++bi) {
        int row = (threadID * 2) + (bi & 0x1);
        row += (bi / 2) * 8;            // rows 0..31
        int col = groupID;              // N=8 columns
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

// Single tile M16 N8 K32 MMA demonstrating `mma.sp` with bf16 inputs.
// Focuses on load/mma/shuffle wiring rather than tiling.
template<int LOAD_METHOD>
__global__ void mma_sp_m16n8k32_fp32bf16_kernel(
    const __nv_bfloat16* __restrict__ A_compressed, // 16x16 (256 bf16)
    const __nv_bfloat16* __restrict__ B,            // 32x8  (256 bf16)
    float* __restrict__ C,                          // 16x8  (128 floats)
    const uint32_t* __restrict__ E                  // 16x1  (16 uint32s -> 16 regs? No, 1 reg/thread?)
                                                   // Metadata: 16x(32dense) = 16x8groups = 16x32bits.
                                                   // We need to load it into the registers.
) {
    // Thread ID 0..31
    int tid = threadIdx.x;
    
    // Fragments
    int a_frag[4];
    int b_frag[4];
    int e_frag[1];
    float c_frag[4] = {0.0f}; // Accumulator
    
    // Manual Load logic for "row" layout (A) and "col" layout (B) compatible with mma.sp
    // We trust the provided "helper" approach or reverse engineer:
    // A: 4 registers. Each holds 2 bf16? No, int = 2 bf16. 4 regs = 8 bf16.
    // 32 threads * 8 bf16 = 256 bf16. Matches 16x16.
    // Mapping for .row:
    // Threads 0..31 cover the matrix.
    // Reference: PTX ISA 9.7.13.4.1. Sparse Matrix Multiply.
    // A (m16k32, row): 
    //   Lane 0..3: Row 0..3? Complex swizzle.
    //   Usually we copy from SMEM using `ldmatrix.sync.aligned.m8n8.x4`.
    
    // We will use Shared Memory to facilitate loading.
    __shared__ __nv_bfloat16 smem_A[16 * 16];
    __shared__ __nv_bfloat16 smem_B[32 * 8];
    __shared__ uint32_t smem_E[16];
    
    // Load data cooperatively to SMEM
    // A: 256 elems. 32 threads. 8 per thread.
    for(int i=0; i<8; ++i) smem_A[tid*8 + i] = A_compressed[tid*8 + i];
    // B: 256 elems.
    for(int i=0; i<8; ++i) smem_B[tid*8 + i] = B[tid*8 + i];
    // E: 16 elems. Thread 0..15 load 1.
    if (tid < 16) smem_E[tid] = E[tid];
    
    __syncthreads();

    // Raw encoding: no shared-memory swap required
    
    // Load from SMEM to Registers
    // A: 16x16. Need 4 regs.
    // B: 32x8. Need 4 regs.
    uint32_t smem_ptr_A = static_cast<uint32_t>(__cvta_generic_to_shared(smem_A));
    uint32_t smem_ptr_B = static_cast<uint32_t>(__cvta_generic_to_shared(smem_B));

    if (LOAD_METHOD == LOAD_METHOD_LDMATRIX_REMAP) {
        // ldmatrix-based load with address remap for raw encoding
        int row = tid % 16;
        int col = (tid / 16) * 8;
        if (row < 8 && col == 8) {
            row += 8;
            col = 0;
        } else if (row >= 8 && col == 0) {
            row -= 8;
            col = 8;
        }
        uint32_t addr_A = smem_ptr_A + row * 16 * sizeof(__nv_bfloat16) + col * sizeof(__nv_bfloat16);
        uint32_t addr_B = smem_ptr_B + tid * 8 * sizeof(__nv_bfloat16); // Row tid.
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                     : "r"(addr_A));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                     : "r"(addr_B));
    } else {
        // Manual fragment load using PTX-documented lane mapping
        load_a_frag_manual(tid, smem_A, a_frag);
        load_b_frag_manual(tid, smem_B, b_frag);
    }

    // E Metadata Load (per-lane mapping for f=0)
    e_frag[0] = load_metadata_for_lane(tid, smem_E);
    
    // MMA
    if (LOAD_METHOD == LOAD_METHOD_LDMATRIX_REMAP) {
        mma_sp_sync_f32_bf16(c_frag, a_frag, b_frag, c_frag, e_frag);
    } else {
        mma_sp_sync_f32_bf16_a0123(c_frag, a_frag, b_frag, c_frag, e_frag);
    }
    
    // Store C (Linear)
    
    __syncthreads();
    
    int gid = tid / 4;
    int row_base = gid;
    
    for (int r = 0; r < 4; ++r) {
        int row = (r < 2) ? row_base : row_base + 8;
        int col = (tid % 4) * 2 + (r % 2);
        
        if (row < 16 && col < 8) {
            C[row * 8 + col] = c_frag[r];
        }
    }
}

// Debug helper
void print_matrix(const char* name, const float* data, int rows, int cols) {
    std::cout << name << ":" << std::endl;
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            std::cout << std::setw(6) << std::setprecision(2) << data[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int m = 16, n = 8, k = 32;
    
    std::vector<__nv_bfloat16> h_A_dense(m * k);
    std::vector<__nv_bfloat16> h_B(k * n);
    std::vector<float> h_C(m * n);
    
    // Random Init
    std::mt19937 gen(42);
    init_structured_sparse_A(h_A_dense, m, k, gen);
    init_random_matrix(h_B, k, n, gen);
    
    // Compress
    std::vector<__nv_bfloat16> h_A_sparse;
    std::vector<uint32_t> h_E;
    compress_matrix_host(h_A_dense, h_A_sparse, h_E, m, k);
    
    // GPU Run
    std::vector<float> h_C_gpu_ldm;
    std::vector<float> h_C_gpu_manual;
    run_gpu_demo(h_A_sparse, h_B, h_E, m, n, h_C_gpu_ldm, h_C_gpu_manual);
    
    // CPU Reference
    std::vector<float> h_C_ref;
    cpu_gemm_ref(h_A_dense, h_B, h_C_ref, m, n, k);
    
    int err_ldm = verify_result("GPU LDMATRIX_REMAP", h_C_gpu_ldm, h_C_ref, m, n);
    int err_manual = verify_result("GPU MANUAL_LANE", h_C_gpu_manual, h_C_ref, m, n);

    return (err_ldm > 0 || err_manual > 0) ? 1 : 0;
}
