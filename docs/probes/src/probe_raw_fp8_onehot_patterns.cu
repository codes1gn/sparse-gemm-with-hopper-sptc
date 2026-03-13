#include "wgmma_sp_raw_common.hpp"

#include <string>

static void init_structured_sparse_A_pattern(std::vector<__nv_fp8_e4m3>& A, int m, int k, int i0, int i1) {
  A.assign(m * k, __nv_fp8_e4m3(0.0f));
  for (int r = 0; r < m; ++r) {
    for (int g = 0; g < k / 4; ++g) {
      int base = r * k + g * 4;
      A[base + i0] = __nv_fp8_e4m3(1.0f + 0.125f * r + 0.03125f * g);
      A[base + i1] = __nv_fp8_e4m3(-0.5f - 0.0625f * r + 0.015625f * g);
    }
  }
}

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

static bool run_case(const std::string& tag, int i0, int i1, int k_offset) {
  constexpr int m = 64;
  constexpr int n = 8;
  constexpr int k = 64;

  std::vector<__nv_fp8_e4m3> a_dense;
  std::vector<__nv_fp8_e4m3> b;
  init_structured_sparse_A_pattern(a_dense, m, k, i0, i1);
  init_onehot_B(b, k, n, k_offset, 1.0f);

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

  wgmma_sp_raw_kernel<__nv_fp8_e4m3, 8><<<dim3(1, 1), dim3(128)>>>(d_a, d_b, d_e, d_c, n, k);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> c_gpu(c_ref.size(), 0.0f);
  CHECK_CUDA(cudaMemcpy(c_gpu.data(), d_c, sizeof(float) * c_gpu.size(), cudaMemcpyDeviceToHost));
  int errors = verify_result(tag, c_gpu, c_ref, n, 1.0e-3f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_e));
  CHECK_CUDA(cudaFree(d_c));
  return errors == 0;
}

static bool run_alt_case(const std::string& tag, int k_offset) {
  constexpr int m = 64;
  constexpr int n = 8;
  constexpr int k = 64;

  std::vector<__nv_fp8_e4m3> a_dense;
  std::vector<__nv_fp8_e4m3> b;
  init_structured_sparse_A_alternating(a_dense, m, k);
  init_onehot_B(b, k, n, k_offset, 1.0f);

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

  wgmma_sp_raw_kernel<__nv_fp8_e4m3, 8><<<dim3(1, 1), dim3(128)>>>(d_a, d_b, d_e, d_c, n, k);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> c_gpu(c_ref.size(), 0.0f);
  CHECK_CUDA(cudaMemcpy(c_gpu.data(), d_c, sizeof(float) * c_gpu.size(), cudaMemcpyDeviceToHost));
  int errors = verify_result(tag, c_gpu, c_ref, n, 1.0e-3f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_e));
  CHECK_CUDA(cudaFree(d_c));
  return errors == 0;
}

int main() {
  int device = select_best_device();
  if (device < 0) return 0;

  bool ok = true;
  ok &= run_case("A01 onehot k0", 0, 1, 0);
  ok &= run_case("A02 onehot k0", 0, 2, 0);
  ok &= run_case("A03 onehot k0", 0, 3, 0);
  ok &= run_case("A12 onehot k0", 1, 2, 0);
  ok &= run_case("A13 onehot k0", 1, 3, 0);
  ok &= run_case("A23 onehot k0", 2, 3, 0);
  ok &= run_case("A01 onehot k8", 0, 1, 8);
  ok &= run_case("A01 onehot k16", 0, 1, 16);
  ok &= run_case("A01 onehot k24", 0, 1, 24);
  ok &= run_case("A01 onehot k32", 0, 1, 32);
  ok &= run_case("A01 onehot k40", 0, 1, 40);
  ok &= run_case("A01 onehot k48", 0, 1, 48);
  ok &= run_case("A01 onehot k56", 0, 1, 56);
  ok &= run_alt_case("ALT onehot k0", 0);
  ok &= run_alt_case("ALT onehot k8", 8);
  ok &= run_alt_case("ALT onehot k16", 16);
  ok &= run_alt_case("ALT onehot k24", 24);
  ok &= run_alt_case("ALT onehot k32", 32);
  ok &= run_alt_case("ALT onehot k40", 40);
  ok &= run_alt_case("ALT onehot k48", 48);
  ok &= run_alt_case("ALT onehot k56", 56);
  return ok ? 0 : 1;
}
