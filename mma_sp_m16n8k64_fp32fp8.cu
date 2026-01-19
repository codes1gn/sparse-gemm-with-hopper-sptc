
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>


#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define LOAD_METHOD_VECTOR 1
#define LOAD_METHOD_SCALAR 2

static constexpr int M = 16;
static constexpr int N = 8;
static constexpr int K = 64;

static void init_structured_sparse_A_f32(std::vector<float>& A, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> pick(0, 3);

  A.assign(M * K, 0.0f);
  for (int r = 0; r < M; ++r) {
    for (int g = 0; g < K / 4; ++g) {
      int i0 = pick(gen);
      int i1 = pick(gen);
      while (i1 == i0) i1 = pick(gen);
      if (i0 > i1) std::swap(i0, i1);

      float v0 = dist(gen) >= 0.0f ? 1.0f : -1.0f;
      float v1 = dist(gen) >= 0.0f ? 1.0f : -1.0f;

      int base = r * K + g * 4;
      A[base + i0] = v0;
      A[base + i1] = v1;
    }
  }
}

static void init_structured_sparse_A_pattern(std::vector<float>& A, int i0, int i1, bool neg) {
  A.assign(M * K, 0.0f);
  for (int r = 0; r < M; ++r) {
    for (int g = 0; g < K / 4; ++g) {
      int base = r * K + g * 4;
      float v0 = 1.0f + 0.1f * r + 0.01f * g;
      float v1 = 2.0f + 0.1f * r + 0.01f * g;
      if (neg) {
        v0 = -v0;
        v1 = -v1;
      }
      A[base + i0] = v0;
      A[base + i1] = v1;
    }
  }
}

static void init_random_B_f32_colmajor(std::vector<float>& B, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  B.resize(K * N);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      float v = dist(gen);
      if (std::abs(v) < 0.25f) v = (v < 0.0f ? -0.25f : 0.25f);
      B[n * K + k] = v;
    }
  }
}

static void init_random_B_f32_colmajor_discrete(std::vector<float>& B, std::mt19937& gen) {
  static const float vals[] = { -2.0f, -1.5f, -1.0f, -0.5f, 0.5f, 1.0f, 1.5f, 2.0f };
  std::uniform_int_distribution<int> pick(0, 7);
  B.resize(K * N);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      B[n * K + k] = vals[pick(gen)];
    }
  }
}

static void init_sparse_B_f32_colmajor(std::vector<float>& B, std::mt19937& gen, int nnz_per_col) {
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<int> pick_k(0, K - 1);
  B.assign(K * N, 0.0f);
  for (int n = 0; n < N; ++n) {
    for (int t = 0; t < nnz_per_col; ++t) {
      int k = pick_k(gen);
      float v = dist(gen);
      if (std::abs(v) < 0.25f) v = (v < 0.0f ? -0.25f : 0.25f);
      B[n * K + k] = v;
    }
  }
}

static void init_onehot_B_f32_colmajor(std::vector<float>& B, int k_offset, float value) {
  B.assign(K * N, 0.0f);
  for (int n = 0; n < N; ++n) {
    int k = k_offset + n;
    if (k >= 0 && k < K) {
      B[n * K + k] = value;
    }
  }
}

static void init_twohot_B_f32_colmajor(std::vector<float>& B, int k0, int k1, float value) {
  B.assign(K * N, 0.0f);
  for (int n = 0; n < N; ++n) {
    if (k0 >= 0 && k0 < K) B[n * K + k0] = value;
    if (k1 >= 0 && k1 < K) B[n * K + k1] = value;
  }
}

template <typename Fp8T>
static void quantize_fp8(const std::vector<float>& in, std::vector<Fp8T>& out) {
  out.resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) out[i] = Fp8T(in[i]);
}

template <typename Fp8A, typename Fp8B>
static void cpu_gemm_ref_fp8(
    const std::vector<Fp8A>& A_fp8,
    const std::vector<Fp8B>& B_fp8_colmajor,
    std::vector<float>& C_ref) {
  C_ref.assign(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a = static_cast<float>(A_fp8[m * K + k]);
        float b = static_cast<float>(B_fp8_colmajor[n * K + k]);
        sum += a * b;
      }
      C_ref[m * N + n] = sum;
    }
  }
}

static void compress_matrix_host_fp8_k64(
  const std::vector<float>& A_dense_f32,
  std::vector<__nv_fp8_e5m2>& A_sparse,
  std::vector<uint32_t>& E_metadata,
  bool sort_indices) {
  constexpr int K_SPARSE = K / 2; // 32
  A_sparse.assign(M * K_SPARSE, __nv_fp8_e5m2(0.0f));
  E_metadata.assign(M * 2, 0); // two 32-bit metadata per row (cols 0-31, 32-63)

  for (int r = 0; r < M; ++r) {
    for (int c_group = 0; c_group < K / 4; ++c_group) {
      int base = r * K + c_group * 4;
      int idxs[2] = {-1, -1};
      int nz = 0;
      for (int i = 0; i < 4; ++i) {
        float v = A_dense_f32[base + i];
        if (v != 0.0f) {
          if (nz < 2) idxs[nz] = i;
          nz++;
        }
      }
      if (sort_indices && idxs[0] > idxs[1]) std::swap(idxs[0], idxs[1]);

      int sparse_col_base = c_group * 2;
      auto store_sparse = [&](int row, int sparse_col, float v) {
        int idx = row * (K / 2) + sparse_col;
        A_sparse[idx] = __nv_fp8_e5m2(v);
      };
      store_sparse(r, sparse_col_base + 0, A_dense_f32[base + idxs[0]]);
      store_sparse(r, sparse_col_base + 1, A_dense_f32[base + idxs[1]]);

      int pack_col = c_group / 8;           // 0 for cols 0-31, 1 for cols 32-63
      int pair_idx = (c_group % 8) * 2;     // 0..14
      uint32_t packed = E_metadata[r * 2 + pack_col];
      packed |= (static_cast<uint32_t>(idxs[0]) << (pair_idx * 2));
      packed |= (static_cast<uint32_t>(idxs[1]) << ((pair_idx + 1) * 2));
      E_metadata[r * 2 + pack_col] = packed;
    }
  }
}

__device__ __forceinline__ uint32_t pack_u8x4(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
  return (static_cast<uint32_t>(b3) << 24) | (static_cast<uint32_t>(b2) << 16) |
         (static_cast<uint32_t>(b1) << 8) | static_cast<uint32_t>(b0);
}

__device__ __forceinline__ void mma_sp_sync_f32_e5m2_e4m3_k64(
    float* d,
    const uint32_t* a,
    const uint32_t* b,
    const float* c,
    uint32_t e) {
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9, %10, %11}, "
      "{%12, %13, %14, %15}, %16, 0;\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        "r"(e));
}

template <int LOAD_METHOD>
__device__ __forceinline__ void build_fragments_m16n8k64_fp8(
    const uint8_t* smem_A,
    const uint8_t* smem_B,
  const uint32_t* smem_E,
    uint32_t* a_frag,
    uint32_t* b_frag,
    uint32_t* e_out) {
  int lane = threadIdx.x & 31;
  int group_id = lane >> 2;
  int thread_id_in_group = lane & 0x3;
  auto load_A_u8 = [&](int row, int col) -> uint8_t {
    int idx = row * (K / 2) + col;
    return smem_A[idx];
  };
  auto load_B_u8 = [&](int n, int k) -> uint8_t { return smem_B[n * K + k]; };

  int rows_A[2] = {group_id, group_id + 8};
  int base_cols_A[2] = {thread_id_in_group * 8, thread_id_in_group * 8 + 32};

  int a_reg = 0;
  uint8_t vals[16];
  #pragma unroll
  for (int ai = 0; ai < 16; ++ai) {
    int row = (ai < 4 || (ai >= 8 && ai < 12)) ? rows_A[0] : rows_A[1];
    int col_base = (ai < 8) ? base_cols_A[0] : base_cols_A[1];
    int chunk = (col_base / 4) + ((ai & 0x2) ? 1 : 0);
    int val_idx = ai & 0x1;
    int sparse_col = chunk * 2 + val_idx;
    vals[ai] = load_A_u8(row, sparse_col);
  }

  a_frag[a_reg++] = pack_u8x4(vals[0], vals[1], vals[2], vals[3]);
  a_frag[a_reg++] = pack_u8x4(vals[4], vals[5], vals[6], vals[7]);
  a_frag[a_reg++] = pack_u8x4(vals[8], vals[9], vals[10], vals[11]);
  a_frag[a_reg++] = pack_u8x4(vals[12], vals[13], vals[14], vals[15]);

  int origin_row_b = (lane & 0x3) * 4;
  int origin_col_b = lane >> 2;
  #pragma unroll
  for (int inner_idx = 0; inner_idx < 4; ++inner_idx) {
    int row = origin_row_b + inner_idx * 16;
    const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&smem_B[origin_col_b * K + row]);
    b_frag[inner_idx] = *ptr;
  }

  // Metadata layout per PTX figure (cols 0-31 and 32-63 split across lanes).
  int half_meta = (thread_id_in_group < 2) ? 0 : 1;
  uint32_t e0 = smem_E[group_id * 2 + half_meta];
  uint32_t e1 = smem_E[(group_id + 8) * 2 + half_meta];
  uint32_t lo0 = e0 & 0xFFFFu;
  uint32_t hi0 = (e0 >> 16) & 0xFFFFu;
  uint32_t lo1 = e1 & 0xFFFFu;
  uint32_t hi1 = (e1 >> 16) & 0xFFFFu;
  if ((thread_id_in_group & 0x1) == 0) {
    *e_out = (lo1 << 16) | lo0;
  } else {
    *e_out = (hi1 << 16) | hi0;
  }
}

template <int LOAD_METHOD>
__global__ void mma_sp_m16n8k64_fp32_e5m2_e4m3_kernel(
  const __nv_fp8_e5m2* __restrict__ A_sparse,
  const __nv_fp8_e4m3* __restrict__ B_colmajor,
  float* __restrict__ C_out,
  const uint32_t* __restrict__ E) {
  int lane = threadIdx.x & 31;
  int group_id = lane >> 2;             // 0..7
  int thread_id_in_group = lane & 0x3;  // 0..3

  __shared__ uint32_t smem_A_u32[(M * (K / 2)) / 4];
  __shared__ uint32_t smem_B_u32[(K * N) / 4];
  __shared__ uint32_t smem_E[M * 2];

  const uint32_t* A_u32 = reinterpret_cast<const uint32_t*>(A_sparse);
  const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B_colmajor);

  for (int i = lane; i < (M * (K / 2)) / 4; i += 32) smem_A_u32[i] = A_u32[i];
  for (int i = lane; i < (K * N) / 4; i += 32) smem_B_u32[i] = B_u32[i];
  for (int i = lane; i < M * 2; i += 32) smem_E[i] = E[i];
  __syncthreads();

  const uint8_t* smem_A = reinterpret_cast<const uint8_t*>(smem_A_u32);
  const uint8_t* smem_B = reinterpret_cast<const uint8_t*>(smem_B_u32);

  uint32_t a_frag[4];
  uint32_t b_frag[4];
  uint32_t e = 0;
  float c_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float d_frag[4];

  build_fragments_m16n8k64_fp8<LOAD_METHOD>(smem_A, smem_B, smem_E, a_frag, b_frag, &e);
  mma_sp_sync_f32_e5m2_e4m3_k64(d_frag, a_frag, b_frag, c_frag, e);

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int out_row = (i < 2) ? group_id : (group_id + 8);
    int out_col = (thread_id_in_group * 2) + (i & 0x1);
    C_out[out_row * N + out_col] = d_frag[i];
  }
}

__global__ void gpu_ref_gemm_fp8_kernel(
    const __nv_fp8_e5m2* __restrict__ A_dense,
    const __nv_fp8_e4m3* __restrict__ B_colmajor,
    float* __restrict__ C_out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          float a = static_cast<float>(A_dense[m * K + k]);
          float b = static_cast<float>(B_colmajor[n * K + k]);
          sum += a * b;
        }
        C_out[m * N + n] = sum;
      }
    }
  }
}

template <typename Fp8T>
__global__ void quantize_fp8_kernel(const float* in, Fp8T* out, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = Fp8T(in[idx]);
  }
}


static void run_one(const char* name) {
  std::mt19937 gen(123);

  auto run_case = [&](const char* tag, auto initA, auto initB, bool sort_indices, bool allow_fail, bool run_gpu_ref) -> bool {
    std::vector<float> A_f32;
    std::vector<float> B_f32_colmajor;
    initA(A_f32);
    initB(B_f32_colmajor);

    std::vector<float> C_ref;
    std::vector<float> C_out(M * N, 0.0f);

    std::vector<__nv_fp8_e5m2> A_fp8;
    std::vector<__nv_fp8_e4m3> B_fp8;
    quantize_fp8(A_f32, A_fp8);
    quantize_fp8(B_f32_colmajor, B_fp8);
    cpu_gemm_ref_fp8(A_fp8, B_fp8, C_ref);

    std::vector<__nv_fp8_e5m2> A_sparse;
    std::vector<uint32_t> E_metadata;
    compress_matrix_host_fp8_k64(A_f32, A_sparse, E_metadata, sort_indices);

    __nv_fp8_e5m2* dA = nullptr;
    __nv_fp8_e5m2* dA_dense = nullptr;
    __nv_fp8_e4m3* dB = nullptr;
    float* dC = nullptr;
    float* dC_ref = nullptr;
    uint32_t* dE = nullptr;
    float* dA_f32 = nullptr;
    float* dB_f32 = nullptr;
    __nv_fp8_e5m2* dA_q = nullptr;
    __nv_fp8_e4m3* dB_q = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeof(__nv_fp8_e5m2) * A_sparse.size()));
    CHECK_CUDA(cudaMalloc(&dA_dense, sizeof(__nv_fp8_e5m2) * A_fp8.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(__nv_fp8_e4m3) * B_fp8.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C_out.size()));
    if (run_gpu_ref) CHECK_CUDA(cudaMalloc(&dC_ref, sizeof(float) * C_out.size()));
    CHECK_CUDA(cudaMalloc(&dE, sizeof(uint32_t) * E_metadata.size()));
    if (run_gpu_ref) {
      CHECK_CUDA(cudaMalloc(&dA_f32, sizeof(float) * A_f32.size()));
      CHECK_CUDA(cudaMalloc(&dB_f32, sizeof(float) * B_f32_colmajor.size()));
      CHECK_CUDA(cudaMalloc(&dA_q, sizeof(__nv_fp8_e5m2) * A_fp8.size()));
      CHECK_CUDA(cudaMalloc(&dB_q, sizeof(__nv_fp8_e4m3) * B_fp8.size()));
    }

    CHECK_CUDA(cudaMemcpy(dA, A_sparse.data(), sizeof(__nv_fp8_e5m2) * A_sparse.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_dense, A_fp8.data(), sizeof(__nv_fp8_e5m2) * A_fp8.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B_fp8.data(), sizeof(__nv_fp8_e4m3) * B_fp8.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_out.size()));
    if (run_gpu_ref) CHECK_CUDA(cudaMemset(dC_ref, 0, sizeof(float) * C_out.size()));
    CHECK_CUDA(cudaMemcpy(dE, E_metadata.data(), sizeof(uint32_t) * E_metadata.size(), cudaMemcpyHostToDevice));

    if (run_gpu_ref) {
      CHECK_CUDA(cudaMemcpy(dA_f32, A_f32.data(), sizeof(float) * A_f32.size(), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(dB_f32, B_f32_colmajor.data(), sizeof(float) * B_f32_colmajor.size(), cudaMemcpyHostToDevice));
      int threads = 256;
      int blocksA = (M * K + threads - 1) / threads;
      int blocksB = (K * N + threads - 1) / threads;
      quantize_fp8_kernel<<<blocksA, threads>>>(dA_f32, dA_q, M * K);
      quantize_fp8_kernel<<<blocksB, threads>>>(dB_f32, dB_q, K * N);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());

      CHECK_CUDA(cudaMemcpy(A_fp8.data(), dA_q, sizeof(__nv_fp8_e5m2) * A_fp8.size(), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(B_fp8.data(), dB_q, sizeof(__nv_fp8_e4m3) * B_fp8.size(), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(dA_dense, dA_q, sizeof(__nv_fp8_e5m2) * A_fp8.size(), cudaMemcpyDeviceToDevice));
      CHECK_CUDA(cudaMemcpy(dB, dB_q, sizeof(__nv_fp8_e4m3) * B_fp8.size(), cudaMemcpyDeviceToDevice));
    }

    dim3 block(32);
    dim3 grid(1);

    mma_sp_m16n8k64_fp32_e5m2_e4m3_kernel<LOAD_METHOD_VECTOR><<<grid, block>>>(dA, dB, dC, dE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (run_gpu_ref) {
      gpu_ref_gemm_fp8_kernel<<<1, 1>>>(dA_dense, dB, dC_ref);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(C_out.data(), dC, sizeof(float) * C_out.size(), cudaMemcpyDeviceToHost));
    std::vector<float> C_ref_gpu;
    if (run_gpu_ref) {
      C_ref_gpu.resize(M * N);
      CHECK_CUDA(cudaMemcpy(C_ref_gpu.data(), dC_ref, sizeof(float) * C_out.size(), cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_dense));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    if (run_gpu_ref) CHECK_CUDA(cudaFree(dC_ref));
    CHECK_CUDA(cudaFree(dE));
    if (run_gpu_ref) {
      CHECK_CUDA(cudaFree(dA_f32));
      CHECK_CUDA(cudaFree(dB_f32));
      CHECK_CUDA(cudaFree(dA_q));
      CHECK_CUDA(cudaFree(dB_q));
    }

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int bad = 0;
    for (int i = 0; i < M * N; ++i) {
      float ref = C_ref[i];
      float got = C_out[i];
      float abs_err = std::abs(ref - got);
      float rel_err = abs_err / (std::abs(ref) + 1e-3f);
      if (abs_err > max_abs_err) max_abs_err = abs_err;
      if (rel_err > max_rel_err) max_rel_err = rel_err;
      if (rel_err > 5e-2f && abs_err > 5e-2f) bad++;
    }

    std::cout << tag << " max_abs_err=" << max_abs_err
              << " max_rel_err=" << max_rel_err
              << " bad=" << bad << std::endl;

    if (run_gpu_ref) {
      int bad_gpu = 0;
      float max_abs_err_gpu = 0.0f;
      for (int i = 0; i < M * N; ++i) {
        float ref = C_ref_gpu[i];
        float got = C_out[i];
        float abs_err = std::abs(ref - got);
        if (abs_err > max_abs_err_gpu) max_abs_err_gpu = abs_err;
        if (abs_err > 5e-2f) bad_gpu++;
      }
      std::cout << tag << " gpu_ref max_abs_err=" << max_abs_err_gpu
                << " bad=" << bad_gpu << std::endl;
    }

    if (bad != 0) {
      std::cout << "First row C_out: ";
      for (int n = 0; n < N; ++n) {
        std::cout << C_out[n] << " ";
      }
      std::cout << "\nFirst row C_ref: ";
      for (int n = 0; n < N; ++n) {
        std::cout << C_ref[n] << " ";
      }
      std::cout << std::endl;
      if (!allow_fail) {
        exit(EXIT_FAILURE);
      }
      return false;
    }
    return true;
  };

  auto initA_random = [&](std::vector<float>& A) { init_structured_sparse_A_f32(A, gen); };
  auto initB_random = [&](std::vector<float>& B) { init_random_B_f32_colmajor(B, gen); };
  auto initB_random_discrete = [&](std::vector<float>& B) { init_random_B_f32_colmajor_discrete(B, gen); };
  auto initB_sparse2 = [&](std::vector<float>& B) { init_sparse_B_f32_colmajor(B, gen, 2); };
  auto initB_sparse8 = [&](std::vector<float>& B) { init_sparse_B_f32_colmajor(B, gen, 8); };
  auto initA_pattern01 = [&](std::vector<float>& A) { init_structured_sparse_A_pattern(A, 0, 1, false); };
  auto initA_pattern03 = [&](std::vector<float>& A) { init_structured_sparse_A_pattern(A, 0, 3, false); };
  auto initA_pattern23 = [&](std::vector<float>& A) { init_structured_sparse_A_pattern(A, 2, 3, false); };
  auto initA_pattern02 = [&](std::vector<float>& A) { init_structured_sparse_A_pattern(A, 0, 2, false); };
  auto initA_pattern12 = [&](std::vector<float>& A) { init_structured_sparse_A_pattern(A, 1, 2, false); };
  auto initA_pattern13 = [&](std::vector<float>& A) { init_structured_sparse_A_pattern(A, 1, 3, false); };
  auto initA_pattern01_neg = [&](std::vector<float>& A) { init_structured_sparse_A_pattern(A, 0, 1, true); };
  for (int offset = 0; offset < K; offset += 8) {
    std::string tag = "pattern A(0,1) + onehot B (k=" + std::to_string(offset) + ".." + std::to_string(offset + 7) + ")";
        auto initB_onehot = [=](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, offset, 1.0f); };
    run_case(tag.c_str(), initA_pattern01, initB_onehot, true, false, false);
  }
  run_case("pattern A(2,3) + onehot B (k=0..7)", initA_pattern23,
          [&](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, 0, 1.0f); }, true, false, false);
  run_case("pattern A(0,2) + onehot B (k=0..7)", initA_pattern02,
          [&](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, 0, 1.0f); }, true, false, false);
  run_case("pattern A(0,3) + onehot B (k=0..7)", initA_pattern03,
           [&](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, 0, 1.0f); }, true, false, false);
  run_case("pattern A(1,2) + onehot B (k=0..7)", initA_pattern12,
           [&](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, 0, 1.0f); }, true, false, false);
  run_case("pattern A(1,3) + onehot B (k=0..7)", initA_pattern13,
           [&](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, 0, 1.0f); }, true, false, false);
      run_case("pattern A(0,1) neg + onehot B (k=0..7)", initA_pattern01_neg,
          [&](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, 0, 1.0f); }, true, false, false);
      run_case("pattern A(0,1) + onehot B -1 (k=0..7)", initA_pattern01,
          [&](std::vector<float>& B) { init_onehot_B_f32_colmajor(B, 0, -1.0f); }, true, false, false);
      run_case("pattern A(0,1) + twohot B (k=0,1)", initA_pattern01,
               [&](std::vector<float>& B) { init_twohot_B_f32_colmajor(B, 0, 1, 1.0f); }, true, false, false);
  bool sorted_ok = run_case(name, initA_random, initB_random, true, true, true);
  bool unsorted_ok = run_case("random A/B (unsorted idx)", initA_random, initB_random, false, true, true);
  bool discrete_ok = run_case("random B discrete", initA_random, initB_random_discrete, true, true, true);
  bool sparse2_ok = run_case("random B sparse2", initA_random, initB_sparse2, true, true, true);
  bool sparse8_ok = run_case("random B sparse8", initA_random, initB_sparse8, true, true, true);
  if (!sorted_ok && !unsorted_ok) {
    std::cerr << "Both sorted and unsorted random cases failed." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Running scalar load path..." << std::endl;

  std::vector<float> A_f32;
  std::vector<float> B_f32_colmajor;
  init_structured_sparse_A_f32(A_f32, gen);
  init_random_B_f32_colmajor(B_f32_colmajor, gen);

  std::vector<float> C_ref;
  std::vector<float> C_out(M * N, 0.0f);

  std::vector<__nv_fp8_e5m2> A_fp8;
  std::vector<__nv_fp8_e4m3> B_fp8;
  quantize_fp8(A_f32, A_fp8);
  quantize_fp8(B_f32_colmajor, B_fp8);
  cpu_gemm_ref_fp8(A_fp8, B_fp8, C_ref);

  std::vector<__nv_fp8_e5m2> A_sparse;
  std::vector<uint32_t> E_metadata;
  compress_matrix_host_fp8_k64(A_f32, A_sparse, E_metadata, true);

  __nv_fp8_e5m2* dA = nullptr;
  __nv_fp8_e4m3* dB = nullptr;
  float* dC = nullptr;
  uint32_t* dE = nullptr;

  CHECK_CUDA(cudaMalloc(&dA, sizeof(__nv_fp8_e5m2) * A_sparse.size()));
  CHECK_CUDA(cudaMalloc(&dB, sizeof(__nv_fp8_e4m3) * B_fp8.size()));
  CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C_out.size()));
  CHECK_CUDA(cudaMalloc(&dE, sizeof(uint32_t) * E_metadata.size()));

  CHECK_CUDA(cudaMemcpy(dA, A_sparse.data(), sizeof(__nv_fp8_e5m2) * A_sparse.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, B_fp8.data(), sizeof(__nv_fp8_e4m3) * B_fp8.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * C_out.size()));
  CHECK_CUDA(cudaMemcpy(dE, E_metadata.data(), sizeof(uint32_t) * E_metadata.size(), cudaMemcpyHostToDevice));

  dim3 block(32);
  dim3 grid(1);

  mma_sp_m16n8k64_fp32_e5m2_e4m3_kernel<LOAD_METHOD_VECTOR><<<grid, block>>>(dA, dB, dC, dE);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(C_out.data(), dC, sizeof(float) * C_out.size(), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaFree(dE));
}

int main() {
  run_one("mma.sp.m16n8k64.fp32.e5m2.e4m3");
  std::cout << "OK" << std::endl;
  return 0;
}
