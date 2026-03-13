#ifndef BENCH_COMMON_HPP
#define BENCH_COMMON_HPP

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status__ = (func);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status__)            \
                << " at line " << __LINE__ << std::endl;                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

template <typename T>
inline float elem_to_float(T x);

template <>
inline float elem_to_float<half>(half x) {
  return __half2float(x);
}

template <>
inline float elem_to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 x) {
  return static_cast<float>(x);
}

template <typename T>
inline T float_to_elem(float x);

template <>
inline half float_to_elem<half>(float x) {
  return __float2half_rn(x);
}

template <>
inline __nv_fp8_e4m3 float_to_elem<__nv_fp8_e4m3>(float x) {
  return __nv_fp8_e4m3(x);
}

template <typename Element>
void init_structured_sparse_a(
    std::vector<Element>& a_dense,
    int m,
    int k,
    std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> pick(0, 3);

  a_dense.assign(m * k, float_to_elem<Element>(0.0f));
  for (int row = 0; row < m; ++row) {
    for (int group = 0; group < k / 4; ++group) {
      int i0 = pick(gen);
      int i1 = pick(gen);
      while (i1 == i0) {
        i1 = pick(gen);
      }
      if (i0 > i1) {
        std::swap(i0, i1);
      }

      float v0 = dist(gen);
      float v1 = dist(gen);
      if (std::fabs(v0) < 0.1f) {
        v0 = (v0 < 0.0f ? -0.5f : 0.5f);
      }
      if (std::fabs(v1) < 0.1f) {
        v1 = (v1 < 0.0f ? -0.75f : 0.75f);
      }

      int base = row * k + group * 4;
      a_dense[base + i0] = float_to_elem<Element>(v0);
      a_dense[base + i1] = float_to_elem<Element>(v1);
    }
  }
}

template <typename Element>
void init_random_b(std::vector<Element>& b, int k, int n, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  b.resize(k * n);
  for (int kk = 0; kk < k; ++kk) {
    for (int nn = 0; nn < n; ++nn) {
      float v = dist(gen);
      if (std::fabs(v) < 0.1f) {
        v = (v < 0.0f ? -0.25f : 0.25f);
      }
      b[kk * n + nn] = float_to_elem<Element>(v);
    }
  }
}

template <typename Element>
void compress_structured_sparse_a(
    const std::vector<Element>& a_dense,
    std::vector<Element>& a_sparse,
    std::vector<uint8_t>& e_bytes,
    int m,
    int k) {
  a_sparse.assign(m * (k / 2), float_to_elem<Element>(0.0f));
  e_bytes.assign(m * (k / 8), uint8_t{0});

  for (int row = 0; row < m; ++row) {
    for (int group = 0; group < k / 4; ++group) {
      int base = row * k + group * 4;
      int idxs[2] = {-1, -1};
      int nz = 0;
      for (int i = 0; i < 4; ++i) {
        if (elem_to_float(a_dense[base + i]) != 0.0f) {
          if (nz < 2) {
            idxs[nz] = i;
          }
          ++nz;
        }
      }
      if (nz != 2) {
        std::cerr << "Invalid 2:4 structure at row " << row << ", group " << group << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (idxs[0] > idxs[1]) {
        std::swap(idxs[0], idxs[1]);
      }

      int sparse_col = group * 2;
      a_sparse[row * (k / 2) + sparse_col + 0] = a_dense[base + idxs[0]];
      a_sparse[row * (k / 2) + sparse_col + 1] = a_dense[base + idxs[1]];

      uint8_t nibble = 0;
      if constexpr (std::is_same_v<Element, __nv_fp8_e4m3>) {
        if (idxs[0] == 0 && idxs[1] == 1) {
          nibble = 0x4;
        } else if (idxs[0] == 1 && idxs[1] == 2) {
          nibble = 0x9;
        } else if (idxs[0] == 2 && idxs[1] == 3) {
          nibble = 0xE;
        } else if (idxs[0] == 0 && idxs[1] == 2) {
          nibble = 0x8;
        } else if (idxs[0] == 1 && idxs[1] == 3) {
          nibble = 0xD;
        } else if (idxs[0] == 0 && idxs[1] == 3) {
          nibble = 0xC;
        } else {
          std::cerr << "Invalid fp8 2:4 indices at row " << row << ", group " << group
                    << ": (" << idxs[0] << ", " << idxs[1] << ")" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        nibble = static_cast<uint8_t>(idxs[0] | (idxs[1] << 2));
      }
      int byte_idx = row * (k / 8) + group / 2;
      if ((group & 1) == 0) {
        e_bytes[byte_idx] = nibble;
      } else {
        e_bytes[byte_idx] |= static_cast<uint8_t>(nibble << 4);
      }
    }
  }
}

template <typename ElementA, typename ElementB>
void cpu_gemm_ref(
    const std::vector<ElementA>& a_dense,
    const std::vector<ElementB>& b,
    std::vector<float>& c,
    int m,
    int n,
    int k) {
  c.assign(m * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float accum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        accum += elem_to_float(a_dense[row * k + kk]) * elem_to_float(b[kk * n + col]);
      }
      c[row * n + col] = accum;
    }
  }
}

inline int verify_result(
    const std::string& tag,
    const std::vector<float>& got,
    const std::vector<float>& ref,
    int cols,
    float atol = 1.0e-1f) {
  int errors = 0;
  for (int idx = 0; idx < static_cast<int>(got.size()); ++idx) {
    float diff = std::fabs(got[idx] - ref[idx]);
    if (diff > atol) {
      ++errors;
      if (errors <= 8) {
        int row = idx / cols;
        int col = idx % cols;
        std::cout << tag << " mismatch at (" << row << "," << col << ") gpu=" << got[idx]
                  << " ref=" << ref[idx] << " diff=" << diff << std::endl;
      }
    }
  }
  std::cout << tag << " total errors: " << errors << std::endl;
  return errors;
}

inline int select_best_device() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));

  int best_device = -1;
  size_t best_free_mem = 0;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess || props.major < 9) {
      cudaGetLastError();
      continue;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }
    if (cudaFree(nullptr) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    if (best_device < 0 || free_mem > best_free_mem) {
      best_device = device;
      best_free_mem = free_mem;
    }
  }

  if (best_device < 0) {
    return -1;
  }

  CHECK_CUDA(cudaSetDevice(best_device));
  CHECK_CUDA(cudaFree(nullptr));
  return best_device;
}

#endif
