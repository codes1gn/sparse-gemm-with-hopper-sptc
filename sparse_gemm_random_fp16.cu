
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
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

// Compress dense matrix A (MxK) to Sparse A (Mx(K/2)) and Metadata E (Mx(K/16))
// Assuming K is multiple of 4? No, K must be multiple of 32 for tensor core.
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
    E_metadata.resize(m * meta_cols_packed);
    
    std::vector<uint32_t> meta_uncompressed(m * (k / 2));
    
    for (int r = 0; r < m; ++r) {
        for (int c_group = 0; c_group < k / 4; ++c_group) {
            // Process group of 4 elements: A_dense[r, c_group*4 + 0..3]
            // Pick 2 largest magnitude
            struct ValIdx { float val; int idx; };
            std::vector<ValIdx> group(4);
            for(int i=0; i<4; ++i) {
                int col = c_group * 4 + i;
                group[i] = { std::abs(__half2float(A_dense[r * k + col])), i };
            }
            // Sort descending
            std::sort(group.begin(), group.end(), [](const ValIdx& a, const ValIdx& b){
                return a.val > b.val;
            });
            
            // Keep indices of top 2, sorted ascending
            int idx0 = group[0].idx;
            int idx1 = group[1].idx;
            if (idx0 > idx1) std::swap(idx0, idx1);
            
            // Fill sparse matrix and uncompressed metadata
            int sparse_col_base = c_group * 2;
            
            // Value 0
            A_sparse[r * k_sparse + sparse_col_base + 0] = A_dense[r * k + c_group * 4 + idx0];
            meta_uncompressed[r * k_sparse + sparse_col_base + 0] = idx0;
            
            // Value 1
            A_sparse[r * k_sparse + sparse_col_base + 1] = A_dense[r * k + c_group * 4 + idx1];
            meta_uncompressed[r * k_sparse + sparse_col_base + 1] = idx1;
        }
    }
    
    // Pack into uint32_t
    for (int i = 0; i < E_metadata.size(); ++i) {
        uint32_t packed = 0;
        for (int j = 0; j < 16; ++j) {
            // Metadata format: lower bits first?
            // "The first encoded index is stored in the 2 LSBs"
            uint32_t val = meta_uncompressed[i * 16 + j];
            packed |= (val << (j * 2));
        }
        E_metadata[i] = packed;
    }
}

// ------------------------------------------------------------------------------------------------
// Device Kernel
// ------------------------------------------------------------------------------------------------

// MMA.SP Wrapper
// Using fp16 input, fp32 accumulator.
// Shape: m16n8k32
__device__ __forceinline__ void mma_sp_sync_f32_f16(
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
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[2]), "r"(a[1]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
}

// Variant without A-reg swap (a0,a1,a2,a3)
__device__ __forceinline__ void mma_sp_sync_f32_f16_a0123(
    float* d,
    const int* a,
    const int* b,
    const float* c,
    const int* e
) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
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
    if (threadID == 0) return smem_E[groupID];
    if (threadID == 1) return smem_E[groupID + 8];
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

// SIMPLIFIED KERNEL FOR DEMO (Single M16 N8 K32 MMA)
// This avoids tiling complexity and focuses on the Operator mechanics.
template<int LOAD_METHOD>
__global__ void simple_sparse_mma_demo(
    const half* __restrict__ A_compressed, // 16x16 (256 halfs)
    const half* __restrict__ B,            // 32x8  (256 halfs)
    float* __restrict__ C,                 // 16x8  (128 floats)
    const uint32_t* __restrict__ E         // 16x1  (16 uint32s -> 16 regs? No, 1 reg/thread?)
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
    // A: 4 registers. Each holds 2 halfs? No, int = 2 halfs. 4 regs = 8 halfs.
    // 32 threads * 8 halfs = 256 halfs. Matches 16x16.
    // Mapping for .row:
    // Threads 0..31 cover the matrix.
    // Reference: PTX ISA 9.7.13.4.1. Sparse Matrix Multiply.
    // A (m16k32, row): 
    //   Lane 0..3: Row 0..3? Complex swizzle.
    //   Usually we copy from SMEM using `ldmatrix.sync.aligned.m8n8.x4`.
    
    // We will use Shared Memory to facilitate loading.
    __shared__ half smem_A[16 * 16];
    __shared__ half smem_B[32 * 8];
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
        uint32_t addr_A = smem_ptr_A + row * 16 * sizeof(half) + col * sizeof(half);
        uint32_t addr_B = smem_ptr_B + tid * 8 * sizeof(half); // Row tid.
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
        mma_sp_sync_f32_f16(c_frag, a_frag, b_frag, c_frag, e_frag);
    } else {
        mma_sp_sync_f32_f16_a0123(c_frag, a_frag, b_frag, c_frag, e_frag);
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
    
    std::vector<half> h_A_dense(m * k);
    std::vector<half> h_B(k * n);
    std::vector<float> h_C(m * n);
    
    // Random Init
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for(int r=0; r<m; ++r) {
        for(int c=0; c<k; ++c) {
            // Ensure 2:4 sparsity structure is "possible" to select meaningfully
            // We set 2 values large, 2 small per group
            float val = dist(gen);
            if ((c % 4) < 2) val += 2.0f; // Bias to ensure selection
            h_A_dense[r*k + c] = __float2half(val);
        }
    }
    
    for(int i=0; i<k*n; ++i) h_B[i] = __float2half(dist(gen));
    
    // Compress
    std::vector<half> h_A_sparse;
    std::vector<uint32_t> h_E;
    compress_matrix_host(h_A_dense, h_A_sparse, h_E, m, k);
    
    // GPU Alloc & Run
    half *d_A, *d_B;
    uint32_t *d_E;
    float *d_C_ldm, *d_C_manual;
    CHECK_CUDA(cudaMalloc(&d_A, h_A_sparse.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_E, h_E.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_C_ldm, h_C.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_manual, h_C.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_sparse.data(), h_A_sparse.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, h_E.data(), h_E.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_C_ldm, 0, h_C.size() * sizeof(float)));
    simple_sparse_mma_demo<LOAD_METHOD_LDMATRIX_REMAP><<<1, 32>>>(d_A, d_B, d_C_ldm, d_E);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(d_C_manual, 0, h_C.size() * sizeof(float)));
    simple_sparse_mma_demo<LOAD_METHOD_MANUAL_LANE><<<1, 32>>>(d_A, d_B, d_C_manual, d_E);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<float> h_C_gpu_ldm(m * n);
    std::vector<float> h_C_gpu_manual(m * n);
    CHECK_CUDA(cudaMemcpy(h_C_gpu_ldm.data(), d_C_ldm, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_gpu_manual.data(), d_C_manual, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU Reference
    std::vector<float> h_C_ref(m * n, 0.0f);
    // Reconstruct effective A
    std::vector<float> A_eff(m * k, 0.0f);
    for (int r = 0; r < m; ++r) {
        for (int c_group = 0; c_group < k / 4; ++c_group) {
            // Re-identify selection logic (same as compression)
            // Ideally we expose the mask from compression_host but for now duplicate logic
            struct ValIdx { float val; int idx; };
            std::vector<ValIdx> group(4);
            for(int i=0; i<4; ++i) {
                int col = c_group * 4 + i;
                group[i] = { std::abs(__half2float(h_A_dense[r * k + col])), i };
            }
            std::sort(group.begin(), group.end(), [](const ValIdx& a, const ValIdx& b){ return a.val > b.val; });
            
            bool keep[4] = {false};
            keep[group[0].idx] = true; 
            keep[group[1].idx] = true;
            
            for(int i=0; i<4; ++i) {
                if(keep[i]) A_eff[r * k + c_group * 4 + i] = __half2float(h_A_dense[r * k + c_group * 4 + i]);
            }
        }
    }
    
    // Matmul
    for(int i=0; i<m; ++i) {
        for(int j=0; j<n; ++j) {
            float sum = 0.0f;
            for(int l=0; l<k; ++l) {
                // B is simple K*N layout
                sum += A_eff[i*k + l] * __half2float(h_B[l*n + j]);
            }
            h_C_ref[i*n + j] = sum;
        }
    }
    
    // Verify (Standard Linear)
    auto verify = [&](const char* tag, const std::vector<float>& gpu) {
        int errors = 0;
        for(int i=0; i<m*n; ++i) {
            float diff = std::abs(gpu[i] - h_C_ref[i]);
            if(diff > 0.1f) {
                errors++;
                if (errors < 10) std::cout << tag << " mismatch at " << i << " GPU: " << gpu[i] << " CPU: " << h_C_ref[i] << std::endl;
            }
        }
        std::cout << tag << " total errors: " << errors << std::endl;
        if (errors > 0) {
            print_matrix(tag, gpu.data(), m, n);
        }
        return errors;
    };

    int err_ldm = verify("GPU LDMATRIX_REMAP", h_C_gpu_ldm);
    int err_manual = verify("GPU MANUAL_LANE", h_C_gpu_manual);

    return (err_ldm > 0 || err_manual > 0) ? 1 : 0;
}
