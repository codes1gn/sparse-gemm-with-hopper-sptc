#include "wgmma_sp_raw_common.hpp"

#include <array>
#include <iostream>

static void init_structured_sparse_A_alternating(std::vector<__nv_fp8_e4m3>& A, int m, int k) {
  A.assign(m * k, __nv_fp8_e4m3(0.0f));
  constexpr int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  for (int r = 0; r < m; ++r) {
    for (int g = 0; g < k / 4; ++g) {
      int base = r * k + g * 4;
      int const* pair = pairs[g % 6];
      A[base + pair[0]] = __nv_fp8_e4m3(1.0f + 0.125f * r + 0.03125f * g);
      A[base + pair[1]] = __nv_fp8_e4m3(-0.5f - 0.0625f * r + 0.015625f * g);
    }
  }
}

static void init_onehot_B(std::vector<__nv_fp8_e4m3>& B, int k, int n, int k_offset, float value) {
  B.assign(k * n, __nv_fp8_e4m3(0.0f));
  for (int col = 0; col < n; ++col) {
    int kk = k_offset + col;
    if (kk >= 0 && kk < k) {
      B[kk * n + col] = __nv_fp8_e4m3(value);
    }
  }
}

__device__ inline uint32_t permuted_meta(uint8_t const* smem_e, int tid, int perm_id) {
  int base = e_thread_byte_offset_k64(tid);
  uint8_t b[4] = {smem_e[base + 0], smem_e[base + 1], smem_e[base + 2], smem_e[base + 3]};
  constexpr int perms[24][4] = {
      {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
      {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
      {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0},
      {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
  };
  int const* p = perms[perm_id];
  return uint32_t(b[p[0]]) | (uint32_t(b[p[1]]) << 8) | (uint32_t(b[p[2]]) << 16) | (uint32_t(b[p[3]]) << 24);
}

__device__ inline uint32_t transformed_meta(uint8_t const* smem_e, int tid, int mode) {
  uint32_t e = permuted_meta(smem_e, tid, mode % 24);
  switch (mode / 24) {
    case 0: return e;
    case 1: return ~e;
    case 2: return e ^ 0x55555555u;
    case 3: return e ^ 0xAAAAAAAAu;
    default: return e;
  }
}

template <int BlockN>
__global__ void kernel(
    __nv_fp8_e4m3 const* __restrict__ a_sparse,
    __nv_fp8_e4m3 const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    float* __restrict__ c,
    int n,
    int k,
    int perm_id) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __shared__ RawSharedStorage<BlockN, __nv_fp8_e4m3> shared;
  constexpr int kAccRegs = RawSparseWgmma<BlockN, __nv_fp8_e4m3>::kAccRegs;
  int tid = threadIdx.x;
  uint64_t desc_a = make_gmma_desc(smem_ptr_as_uint(shared.smem_A), 64, 8);
  uint64_t desc_b = make_gmma_desc(smem_ptr_as_uint(shared.smem_B), BlockN, BlockN == 8 ? 0 : 8);
  float accum[kAccRegs];
  #pragma unroll
  for (int i = 0; i < kAccRegs; ++i) accum[i] = 0.0f;

  for (int idx = tid; idx < kRawBlockM * 32; idx += kRawThreads) {
    int row = idx / 32;
    int col = idx % 32;
    shared.smem_A[raw_smem_a_index<__nv_fp8_e4m3>(row, col)] = a_sparse[row * (k / 2) + col];
  }
  for (int idx = tid; idx < BlockN * 64; idx += kRawThreads) {
    int col = idx / 64;
    int kk = idx % 64;
    shared.smem_B[raw_smem_b_index<BlockN, __nv_fp8_e4m3>(col, kk)] = b[kk * n + col];
  }
  for (int idx = tid; idx < kRawBlockM * 8; idx += kRawThreads) {
    int row = idx / 8;
    int byte_col = idx % 8;
    shared.smem_E[raw_smem_e_index<__nv_fp8_e4m3>(row, byte_col)] = e_bytes[row * (k / 8) + byte_col];
  }
  __syncthreads();

  uint32_t e = transformed_meta(shared.smem_E, tid, perm_id);
  warpgroup_fence_accum(accum);
  warpgroup_arrive();
  RawSparseWgmma<BlockN, __nv_fp8_e4m3>::fma(desc_a, desc_b, accum, e, 0);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_accum(accum);
  __syncthreads();
  store_accum<BlockN>(accum, c, n, 0, 0, tid);
#endif
}

static bool run_perm(int perm_id) {
  constexpr int m = 64;
  constexpr int n = 8;
  constexpr int k = 64;
  std::vector<__nv_fp8_e4m3> a_dense;
  std::vector<__nv_fp8_e4m3> b;
  init_structured_sparse_A_alternating(a_dense, m, k);
  init_onehot_B(b, k, n, 0, 1.0f);
  std::vector<__nv_fp8_e4m3> a_sparse;
  std::vector<uint8_t> e_bytes;
  compress_structured_sparse_a(a_dense, a_sparse, e_bytes, m, k);
  std::vector<float> c_ref;
  cpu_gemm_ref(a_dense, b, c_ref, m, n, k);

  __nv_fp8_e4m3* d_a = nullptr;
  __nv_fp8_e4m3* d_b = nullptr;
  uint8_t* d_e = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(__nv_fp8_e4m3) * a_sparse.size()));
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(__nv_fp8_e4m3) * b.size()));
  CHECK_CUDA(cudaMalloc(&d_e, sizeof(uint8_t) * e_bytes.size()));
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * c_ref.size()));
  CHECK_CUDA(cudaMemcpy(d_a, a_sparse.data(), sizeof(__nv_fp8_e4m3) * a_sparse.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), sizeof(__nv_fp8_e4m3) * b.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_e, e_bytes.data(), sizeof(uint8_t) * e_bytes.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float) * c_ref.size()));

  kernel<8><<<1, 128>>>(d_a, d_b, d_e, d_c, n, k, perm_id);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> c_gpu(c_ref.size(), 0.0f);
  CHECK_CUDA(cudaMemcpy(c_gpu.data(), d_c, sizeof(float) * c_gpu.size(), cudaMemcpyDeviceToHost));
  int errors = 0;
  for (size_t i = 0; i < c_gpu.size(); ++i) {
    if (fabs(c_gpu[i] - c_ref[i]) > 1.0e-3f) ++errors;
  }
  std::cout << "perm " << perm_id << " errors=" << errors << std::endl;
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_e));
  CHECK_CUDA(cudaFree(d_c));
  return errors == 0;
}

int main() {
  if (select_best_device() < 0) return 0;
  bool any = false;
  for (int perm = 0; perm < 96; ++perm) {
    any |= run_perm(perm);
  }
  return any ? 0 : 1;
}
