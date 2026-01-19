
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
// 1: ldmatrix (B row-major)
// 2: manual lane loads (B col-major)
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

// K=16: 4 groups per row -> 8 indices (2 bits each) -> 16-bit metadata per row.
// Metadata is interleaved as (row i | row i+8) in a single uint32_t per groupID.
void compress_matrix_host(
    const std::vector<half>& A_dense,
    std::vector<half>& A_sparse,
    std::vector<uint32_t>& E_metadata,
    int m, int k
) {
    int k_sparse = k / 2; // 8
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
                std::cerr << "Invalid 2:4 structure at row " << r
                          << ", group " << c_group << ": nonzeros=" << nz << std::endl;
                exit(EXIT_FAILURE);
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

    E_metadata.assign(8, 0);
    for (int g = 0; g < 8; ++g) {
        uint32_t lo = row_meta[g];
        uint32_t hi = row_meta[g + 8];
        E_metadata[g] = (hi << 16) | lo;
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

__device__ __forceinline__ uint32_t pack_half2(const half lo, const half hi) {
    uint32_t ulo = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&lo));
    uint32_t uhi = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&hi));
    return (uhi << 16) | ulo;
}

// MMA.SP Wrapper (K=16)
__device__ __forceinline__ void mma_sp_sync_f32_f16_k16(
    float* d,
    const uint32_t* a,
    const uint32_t* b,
    const float* c,
    const int* e
) {
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11}, %12, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e[0])
    );
}

template<int LOAD_METHOD>
__global__ void mma_sp_m16n8k16_fp32fp16_kernel(
    const half* __restrict__ A_compressed, // 16x8 (128 half)
    const half* __restrict__ B,            // 16x8 (row-major for ldmatrix, col-major for manual)
    float* __restrict__ C,                 // 16x8
    const uint32_t* __restrict__ E         // 8 metadata entries
) {
    int tid = threadIdx.x;

    uint32_t a_frag[2];
    uint32_t b_frag[2];
    int e_frag[1];
    float c_frag[4] = {0.0f};

    __shared__ half smem_A[16 * 8];
    __shared__ half smem_B[16 * 8];
    __shared__ uint32_t smem_E[8];

    for (int i = 0; i < 4; ++i) smem_A[tid * 4 + i] = A_compressed[tid * 4 + i];
    for (int i = 0; i < 4; ++i) smem_B[tid * 4 + i] = B[tid * 4 + i];
    if (tid < 8) smem_E[tid] = E[tid];
    __syncthreads();

    int group_id = tid / 4;       // 0..7
    int col_base = (tid % 4) * 2; // 0,2,4,6

    if (LOAD_METHOD == LOAD_METHOD_LDMATRIX) {
        uint32_t smem_ptr_A = static_cast<uint32_t>(__cvta_generic_to_shared(smem_A));
        uint32_t addr_A = smem_ptr_A + (tid % 16) * 8 * sizeof(half);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                     : "=r"(a_frag[0]), "=r"(a_frag[1])
                     : "r"(addr_A));

        uint32_t smem_ptr_B = static_cast<uint32_t>(__cvta_generic_to_shared(smem_B));
        uint32_t addr_B = smem_ptr_B + tid * 8 * sizeof(half);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                     : "=r"(b_frag[0]), "=r"(b_frag[1])
                     : "r"(addr_B));
    } else {
        half a0 = smem_A[group_id * 8 + col_base + 0];
        half a1 = smem_A[group_id * 8 + col_base + 1];
        half a2 = smem_A[(group_id + 8) * 8 + col_base + 0];
        half a3 = smem_A[(group_id + 8) * 8 + col_base + 1];
        a_frag[0] = pack_half2(a0, a1);
        a_frag[1] = pack_half2(a2, a3);

        int col_b = tid / 4;          // 0..7
        int row_b_base = (tid % 4) * 2;
        half b0 = smem_B[col_b * 16 + row_b_base + 0];
        half b1 = smem_B[col_b * 16 + row_b_base + 1];
        half b2 = smem_B[col_b * 16 + row_b_base + 8];
        half b3 = smem_B[col_b * 16 + row_b_base + 9];
        b_frag[0] = pack_half2(b0, b1);
        b_frag[1] = pack_half2(b2, b3);
    }

    e_frag[0] = smem_E[group_id];
    mma_sp_sync_f32_f16_k16(c_frag, a_frag, b_frag, c_frag, e_frag);

    for (int r = 0; r < 4; ++r) {
        int row = (r < 2) ? group_id : group_id + 8;
        int col = (tid % 4) * 2 + (r % 2);
        if (row < 16 && col < 8) C[row * 8 + col] = c_frag[r];
    }
}

int verify_result(const char* tag, const std::vector<float>& gpu, const std::vector<float>& ref) {
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
    return errors;
}

int main() {
    int m = 16, n = 8, k = 16;

    std::vector<half> h_A_dense(m * k);
    std::vector<half> h_B(k * n);
    std::vector<float> h_C_ref;

    std::mt19937 gen(42);
    init_structured_sparse_A(h_A_dense, m, k, gen);
    init_random_matrix(h_B, k, n, gen);

    std::vector<half> h_A_sparse;
    std::vector<uint32_t> h_E;
    compress_matrix_host(h_A_dense, h_A_sparse, h_E, m, k);

    std::vector<half> h_B_col(k * n);
    for (int r = 0; r < k; ++r) {
        for (int c = 0; c < n; ++c) {
            h_B_col[c * k + r] = h_B[r * n + c];
        }
    }

    half *d_A, *d_B_row, *d_B_col;
    uint32_t *d_E;
    float *d_C_ldm, *d_C_manual;
    CHECK_CUDA(cudaMalloc(&d_A, h_A_sparse.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_row, h_B.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_col, h_B_col.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_E, h_E.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_C_ldm, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_manual, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A_sparse.data(), h_A_sparse.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_row, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_col, h_B_col.data(), h_B_col.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, h_E.data(), h_E.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_C_ldm, 0, m * n * sizeof(float)));
    mma_sp_m16n8k16_fp32fp16_kernel<LOAD_METHOD_LDMATRIX><<<1, 32>>>(d_A, d_B_row, d_C_ldm, d_E);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(d_C_manual, 0, m * n * sizeof(float)));
    mma_sp_m16n8k16_fp32fp16_kernel<LOAD_METHOD_MANUAL><<<1, 32>>>(d_A, d_B_col, d_C_manual, d_E);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> h_C_gpu_ldm(m * n);
    std::vector<float> h_C_gpu_manual(m * n);
    CHECK_CUDA(cudaMemcpy(h_C_gpu_ldm.data(), d_C_ldm, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_gpu_manual.data(), d_C_manual, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    cpu_gemm_ref(h_A_dense, h_B, h_C_ref, m, n, k);

    int errs_ldm = verify_result("GPU LDMATRIX", h_C_gpu_ldm, h_C_ref);
    int errs_manual = verify_result("GPU MANUAL", h_C_gpu_manual, h_C_ref);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B_row));
    CHECK_CUDA(cudaFree(d_B_col));
    CHECK_CUDA(cudaFree(d_E));
    CHECK_CUDA(cudaFree(d_C_ldm));
    CHECK_CUDA(cudaFree(d_C_manual));

    return (errs_ldm > 0 || errs_manual > 0) ? 1 : 0;
}
