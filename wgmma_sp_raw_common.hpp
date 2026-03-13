#pragma once

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
inline float elem_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
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
inline __nv_bfloat16 float_to_elem<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
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
void init_pattern_sparse_a(std::vector<Element>& a_dense, int m, int k) {
  a_dense.assign(m * k, float_to_elem<Element>(0.0f));
  for (int row = 0; row < m; ++row) {
    for (int group = 0; group < k / 4; ++group) {
      int base = row * k + group * 4;
      float v0 = 1.0f + 0.125f * row + 0.03125f * group;
      float v1 = -0.5f - 0.0625f * row + 0.015625f * group;
      a_dense[base + 0] = float_to_elem<Element>(v0);
      a_dense[base + 2] = float_to_elem<Element>(v1);
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
void init_pattern_b(std::vector<Element>& b, int k, int n) {
  b.resize(k * n);
  for (int kk = 0; kk < k; ++kk) {
    for (int nn = 0; nn < n; ++nn) {
      float v = (static_cast<float>((kk % 7) - 3) * 0.25f) + static_cast<float>(nn + 1) * 0.5f;
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
    float atol = 1.0e-2f) {
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

constexpr int kRawBlockM = 64;
constexpr int kRawThreads = 128;
constexpr int kRawFp8MetadataSmemBytes = kRawBlockM * 8;

template <typename Element>
struct RawSparseConfig;

template <>
struct RawSparseConfig<half> {
  static constexpr int kBlockK = 32;
  static constexpr int kSparseK = kBlockK / 2;
  static constexpr int kMetaBytes = kBlockK / 8;
  static constexpr int kASmemElements = kRawBlockM * kSparseK;
  static constexpr int kESmemBytes = kRawBlockM * kMetaBytes;
  static constexpr uint32_t kADescLeading = 64;
  static constexpr uint32_t kADescStride = 8;
};

template <>
struct RawSparseConfig<__nv_bfloat16> {
  static constexpr int kBlockK = 32;
  static constexpr int kSparseK = kBlockK / 2;
  static constexpr int kMetaBytes = kBlockK / 8;
  static constexpr int kASmemElements = kRawBlockM * kSparseK;
  static constexpr int kESmemBytes = kRawBlockM * kMetaBytes;
  static constexpr uint32_t kADescLeading = 64;
  static constexpr uint32_t kADescStride = 8;
};

template <>
struct RawSparseConfig<__nv_fp8_e4m3> {
  static constexpr int kBlockK = 64;
  static constexpr int kSparseK = kBlockK / 2;
  static constexpr int kMetaBytes = kBlockK / 8;
  static constexpr int kASmemElements = kRawBlockM * kSparseK;
  static constexpr int kESmemBytes = kRawFp8MetadataSmemBytes;
  static constexpr uint32_t kADescLeading = 64;
  static constexpr uint32_t kADescStride = 8;
};

template <int BlockN, typename Element>
struct RawSharedStorage {
  alignas(128) Element smem_A[RawSparseConfig<Element>::kASmemElements];
  alignas(128) Element smem_B[BlockN * RawSparseConfig<Element>::kBlockK];
  alignas(128) uint8_t smem_E[RawSparseConfig<Element>::kESmemBytes];
};

__device__ inline uint32_t smem_ptr_as_uint(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}

__device__ inline uint64_t make_gmma_desc(uint32_t smem_addr, uint32_t leading, uint32_t stride) {
  uint64_t desc = static_cast<uint64_t>((smem_addr >> 4) & 0x3fff);
  desc |= static_cast<uint64_t>(leading) << 16;
  desc |= static_cast<uint64_t>(stride) << 32;
  return desc;
}

__device__ inline void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ inline void warpgroup_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ inline void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int NumRegs>
__device__ inline void warpgroup_fence_accum(float (&regs)[NumRegs]) {
#pragma unroll
  for (int i = 0; i < NumRegs; ++i) {
    asm volatile("" : "+f"(regs[i]) :: "memory");
  }
}

__device__ inline uint32_t ld_shared_u32(void const* ptr) {
  uint32_t value = 0;
  uint32_t addr = smem_ptr_as_uint(ptr);
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(addr));
  return value;
}

__device__ inline int smem_a_index(int row, int col) {
  return (col & 7) + ((row >> 3) * 64) + ((row & 7) * 8) + ((col >> 3) * 512);
}

__device__ inline int smem_a_index_k64_e4m3(int row, int col) {
  return (col & 15) + row * 16 + ((col >> 4) * 1024);
}

template <typename Element>
__device__ inline int raw_smem_a_index(int row, int col) {
  if constexpr (std::is_same_v<Element, __nv_fp8_e4m3>) {
    return smem_a_index_k64_e4m3(row, col);
  } else {
    return smem_a_index(row, col);
  }
}

template <int BlockN>
__device__ inline int smem_b_index(int col, int kk) {
  return (col & 7) + ((col >> 3) * 64) + ((kk >> 3) * (BlockN * 8)) + ((kk & 7) * 8);
}

template <int BlockN>
__device__ inline int smem_b_index_k64_e4m3(int col, int kk) {
  return (kk & 15) + col * 16 + ((kk >> 4) * (BlockN * 16));
}

template <int BlockN, typename Element>
__device__ inline int raw_smem_b_index(int col, int kk) {
  if constexpr (std::is_same_v<Element, __nv_fp8_e4m3>) {
    return smem_b_index_k64_e4m3<BlockN>(col, kk);
  } else {
    return smem_b_index<BlockN>(col, kk);
  }
}

__device__ inline int smem_e_index(int row, int byte_col) {
  return (byte_col & 1) + ((byte_col >> 1) * 32) + ((row >> 4) * 64) + ((row & 7) * 4) + (((row >> 3) & 1) * 2);
}

__device__ inline int smem_e_index_k64(int row, int byte_col) {
  return (byte_col & 1) + ((byte_col >> 1) * 32) + ((row >> 4) * 128) + ((row & 7) * 4) + (((row >> 3) & 1) * 2);
}

__device__ inline int smem_e_index_k64_e4m3(int row, int byte_col) {
  int row_block = row >> 4;
  int row_in_block = row & 15;
  int row_lo = row_in_block & 7;
  int row_hi = row_in_block >> 3;
  int col_group = byte_col >> 2;
  int col_lo = byte_col & 3;
  return row_block * 128 + col_group * 64 + row_lo * 8 + row_hi * 4 + col_lo;
}

__device__ inline int e_thread_byte_offset(int tid) {
  return ((tid & 1) * 32) + ((tid >> 5) * 64) + (((tid >> 2) & 7) * 4);
}

__device__ inline int e_thread_byte_offset_k64(int tid) {
  return ((tid & 3) * 32) + ((tid >> 5) * 128) + (((tid >> 2) & 7) * 4);
}

__device__ inline int e_thread_byte_offset_k64_e4m3(int tid) {
  int row = ((tid >> 2) & 7) + ((tid & 1) << 3) + ((tid >> 5) << 4);
  int byte_col = ((tid >> 1) & 1) << 2;
  return smem_e_index_k64_e4m3(row, byte_col);
}

template <typename Element>
__device__ inline int raw_smem_e_index(int row, int byte_col) {
  if constexpr (std::is_same_v<Element, __nv_fp8_e4m3>) {
    return smem_e_index_k64_e4m3(row, byte_col);
  } else if constexpr (RawSparseConfig<Element>::kBlockK == 64) {
    return smem_e_index_k64(row, byte_col);
  } else {
    return smem_e_index(row, byte_col);
  }
}

template <typename Element>
__device__ inline int raw_e_thread_byte_offset(int tid) {
  if constexpr (std::is_same_v<Element, __nv_fp8_e4m3>) {
    return e_thread_byte_offset_k64_e4m3(tid);
  } else if constexpr (RawSparseConfig<Element>::kBlockK == 64) {
    return e_thread_byte_offset_k64(tid);
  } else {
    return e_thread_byte_offset(tid);
  }
}

template <int BlockN, typename Element>
struct BDescParams {
  static constexpr uint32_t kLeading = BlockN;
  static constexpr uint32_t kStride = BlockN == 8 ? 0u : 8u;
};

template <int BlockN>
struct BDescParams<BlockN, __nv_fp8_e4m3> {
  static constexpr uint32_t kLeading = BlockN;
  static constexpr uint32_t kStride = BlockN == 8 ? 0u : 8u;
};

template <int BlockN, typename Element>
__device__ inline uint32_t raw_metadata_u32(
    uint8_t const* e_bytes,
    uint8_t* smem_e,
    int block_row,
    int k_tile,
    int k,
    int tid) {
  (void)e_bytes;
  (void)block_row;
  (void)k_tile;
  (void)k;
  if constexpr (std::is_same_v<Element, __nv_fp8_e4m3>) {
    return ld_shared_u32(smem_e + raw_e_thread_byte_offset<Element>(tid));
  } else {
    return ld_shared_u32(smem_e + raw_e_thread_byte_offset<Element>(tid));
  }
}

template <int BlockN, typename Element>
struct RawSparseWgmma;

#define RAW_OUT_FLOATS_32                                                     \
  "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),                    \
      "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),                \
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),              \
      "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),            \
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),            \
      "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),            \
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),            \
      "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])

#define RAW_OUT_FLOATS_64                                                     \
  RAW_OUT_FLOATS_32, "+f"(d[32]), "+f"(d[33]), "+f"(d[34]),          \
      "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]),            \
      "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]),            \
      "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]),            \
      "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]),            \
      "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]),            \
      "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]),            \
      "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]),            \
      "+f"(d[63])

#define RAW_OUT_FLOATS_128                                                    \
  RAW_OUT_FLOATS_64, "+f"(d[64]), "+f"(d[65]), "+f"(d[66]),          \
      "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]),            \
      "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]),            \
      "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]),            \
      "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]),            \
      "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]),            \
      "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]),            \
      "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]),            \
      "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]),            \
      "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]),         \
      "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]),        \
      "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]),        \
      "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]),        \
      "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]),        \
      "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]),        \
      "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]),        \
      "+f"(d[127])

template <>
struct RawSparseWgmma<8, half> {
  static constexpr int kAccRegs = 4;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %8, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.f16.f16 "
        "{%0, %1, %2, %3}, %4, %5, %6, %7, p, %9, %10, %11, %12;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<16, half> {
  static constexpr int kAccRegs = 8;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %12, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n16k32.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, %10, %11, p, %13, %14, %15, %16;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<32, half> {
  static constexpr int kAccRegs = 16;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %20, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n32k32.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, %18, %19, p, %21, %22, %23, %24;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<64, half> {
  static constexpr int kAccRegs = 32;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %36, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n64k32.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, %34, %35, p, %37, %38, %39, %40;\n"
        "}\n"
        : RAW_OUT_FLOATS_32
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<128, half> {
  static constexpr int kAccRegs = 64;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %68, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, %66, %67, p, %69, %70, %71, %72;\n"
        "}\n"
        : RAW_OUT_FLOATS_64
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<256, half> {
  static constexpr int kAccRegs = 128;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %132, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n256k32.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, "
        "%72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, "
        "%88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, "
        "%104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, "
        "%120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, %130, %131, p, %133, %134, %135, %136;\n"
        "}\n"
        : RAW_OUT_FLOATS_128
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<8, __nv_bfloat16> {
  static constexpr int kAccRegs = 4;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %8, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %4, %5, %6, %7, p, %9, %10, %11, %12;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<16, __nv_bfloat16> {
  static constexpr int kAccRegs = 8;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %12, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n16k32.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, %10, %11, p, %13, %14, %15, %16;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<32, __nv_bfloat16> {
  static constexpr int kAccRegs = 16;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %20, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n32k32.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, %18, %19, p, %21, %22, %23, %24;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<64, __nv_bfloat16> {
  static constexpr int kAccRegs = 32;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %36, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n64k32.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, %34, %35, p, %37, %38, %39, %40;\n"
        "}\n"
        : RAW_OUT_FLOATS_32
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<128, __nv_bfloat16> {
  static constexpr int kAccRegs = 64;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %68, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, %66, %67, p, %69, %70, %71, %72;\n"
        "}\n"
        : RAW_OUT_FLOATS_64
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<256, __nv_bfloat16> {
  static constexpr int kAccRegs = 128;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %132, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n256k32.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, "
        "%72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, "
        "%88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, "
        "%104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, "
        "%120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, %130, %131, p, %133, %134, %135, %136;\n"
        "}\n"
        : RAW_OUT_FLOATS_128
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
  }
};

template <>
struct RawSparseWgmma<8, __nv_fp8_e4m3> {
  static constexpr int kAccRegs = 4;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %8, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n8k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3}, %4, %5, %6, %7, p, %9, %10;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1));
  }
};

template <>
struct RawSparseWgmma<16, __nv_fp8_e4m3> {
  static constexpr int kAccRegs = 8;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %12, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n16k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, %10, %11, p, %13, %14;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1));
  }
};

template <>
struct RawSparseWgmma<32, __nv_fp8_e4m3> {
  static constexpr int kAccRegs = 16;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %20, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n32k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, %18, %19, p, %21, %22;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1));
  }
};

template <>
struct RawSparseWgmma<64, __nv_fp8_e4m3> {
  static constexpr int kAccRegs = 32;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %36, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n64k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, %34, %35, p, %37, %38;\n"
        "}\n"
        : RAW_OUT_FLOATS_32
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1));
  }
};

template <>
struct RawSparseWgmma<128, __nv_fp8_e4m3> {
  static constexpr int kAccRegs = 64;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %68, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n128k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, %66, %67, p, %69, %70;\n"
        "}\n"
        : RAW_OUT_FLOATS_64
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1));
  }
};

template <>
struct RawSparseWgmma<256, __nv_fp8_e4m3> {
  static constexpr int kAccRegs = 128;

  __device__ static void fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %132, 0;\n"
        "  wgmma.mma_async.sp.sync.aligned.m64n256k64.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, "
        "%72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, "
        "%88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, "
        "%104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, "
        "%120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, %130, %131, p, %133, %134;\n"
        "}\n"
        : RAW_OUT_FLOATS_128
        : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1));
  }
};

#undef RAW_OUT_FLOATS_128
#undef RAW_OUT_FLOATS_64
#undef RAW_OUT_FLOATS_32

template <int BlockN>
__device__ inline void store_accum(float const* accum, float* c, int n, int block_row, int block_col, int tid) {
  int row = block_row + ((tid >> 5) * 16) + ((tid >> 2) & 7);
  int col = block_col + ((tid & 3) * 2);
#pragma unroll
  for (int group = 0; group < BlockN / 8; ++group) {
    int idx = group * 4;
    int col_group = col + group * 8;
    c[row * n + col_group + 0] = accum[idx + 0];
    c[row * n + col_group + 1] = accum[idx + 1];
    c[(row + 8) * n + col_group + 0] = accum[idx + 2];
    c[(row + 8) * n + col_group + 1] = accum[idx + 3];
  }
}

template <typename Element, int BlockN>
__global__ void wgmma_sp_raw_kernel(
    Element const* __restrict__ a_sparse,
    Element const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    float* __restrict__ c,
    int n,
    int k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __shared__ RawSharedStorage<BlockN, Element> shared;

  constexpr int kAccRegs = RawSparseWgmma<BlockN, Element>::kAccRegs;
  constexpr int kBlockK = RawSparseConfig<Element>::kBlockK;
  constexpr int kSparseK = RawSparseConfig<Element>::kSparseK;
  constexpr int kMetaBytes = RawSparseConfig<Element>::kMetaBytes;
  int tid = threadIdx.x;
  int block_row = blockIdx.y * kRawBlockM;
  int block_col = blockIdx.x * BlockN;

  uint64_t desc_a = make_gmma_desc(
      smem_ptr_as_uint(shared.smem_A),
      RawSparseConfig<Element>::kADescLeading,
      RawSparseConfig<Element>::kADescStride);
  uint64_t desc_b = make_gmma_desc(
      smem_ptr_as_uint(shared.smem_B),
      BDescParams<BlockN, Element>::kLeading,
      BDescParams<BlockN, Element>::kStride);

  float accum[kAccRegs];
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    accum[i] = 0.0f;
  }

  bool first_k_tile = true;
  for (int k_tile = 0; k_tile < k; k_tile += kBlockK) {
    for (int idx = tid; idx < kRawBlockM * kSparseK; idx += kRawThreads) {
      int row = idx / kSparseK;
      int col = idx % kSparseK;
      shared.smem_A[raw_smem_a_index<Element>(row, col)] =
          a_sparse[(block_row + row) * (k / 2) + (k_tile / 2) + col];
    }

    for (int idx = tid; idx < BlockN * kBlockK; idx += kRawThreads) {
      int col = idx / kBlockK;
      int kk = idx % kBlockK;
      shared.smem_B[raw_smem_b_index<BlockN, Element>(col, kk)] =
          b[(k_tile + kk) * n + (block_col + col)];
    }

    for (int idx = tid; idx < kRawBlockM * kMetaBytes; idx += kRawThreads) {
      int row = idx / kMetaBytes;
      int byte_col = idx % kMetaBytes;
      shared.smem_E[raw_smem_e_index<Element>(row, byte_col)] =
          e_bytes[(block_row + row) * (k / 8) + (k_tile / 8) + byte_col];
    }

    __syncthreads();

    uint32_t e = raw_metadata_u32<BlockN, Element>(e_bytes, shared.smem_E, block_row, k_tile, k, tid);
    int scale_d = first_k_tile ? 0 : 1;

    warpgroup_fence_accum(accum);
    warpgroup_arrive();
    RawSparseWgmma<BlockN, Element>::fma(desc_a, desc_b, accum, e, scale_d);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_accum(accum);

    __syncthreads();
    first_k_tile = false;
  }

  store_accum<BlockN>(accum, c, n, block_row, block_col, tid);
#endif
}

template <typename Element, int BlockN>
struct RawWgmmaSparseDemo {
  static bool run_case(
      const std::string& tag,
      int m,
      int n,
      int k,
      bool use_pattern,
      std::mt19937& gen) {
    constexpr int kBlockK = RawSparseConfig<Element>::kBlockK;
    if ((m % kRawBlockM) != 0 || (n % BlockN) != 0 || (k % kBlockK) != 0) {
      std::cerr << tag << " requires M%64==0, N%" << BlockN << "==0, K%" << kBlockK << "==0" << std::endl;
      return false;
    }

    std::vector<Element> a_dense;
    std::vector<Element> b;
    if (use_pattern) {
      init_pattern_sparse_a(a_dense, m, k);
      init_pattern_b(b, k, n);
    } else {
      init_structured_sparse_a(a_dense, m, k, gen);
      init_random_b(b, k, n, gen);
    }

    std::vector<Element> a_sparse;
    std::vector<uint8_t> e_bytes;
    compress_structured_sparse_a(a_dense, a_sparse, e_bytes, m, k);

    std::vector<float> c_ref;
    cpu_gemm_ref(a_dense, b, c_ref, m, n, k);

    Element* d_a = nullptr;
    Element* d_b = nullptr;
    uint8_t* d_e = nullptr;
    float* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, sizeof(Element) * a_sparse.size()));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(Element) * b.size()));
    CHECK_CUDA(cudaMalloc(&d_e, sizeof(uint8_t) * e_bytes.size()));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * c_ref.size()));

    CHECK_CUDA(cudaMemcpy(d_a, a_sparse.data(), sizeof(Element) * a_sparse.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), sizeof(Element) * b.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_e, e_bytes.data(), sizeof(uint8_t) * e_bytes.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float) * c_ref.size()));

    dim3 block(kRawThreads);
    dim3 grid(n / BlockN, m / kRawBlockM);
    wgmma_sp_raw_kernel<Element, BlockN><<<grid, block>>>(d_a, d_b, d_e, d_c, n, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> c_gpu(c_ref.size(), 0.0f);
    CHECK_CUDA(cudaMemcpy(c_gpu.data(), d_c, sizeof(float) * c_gpu.size(), cudaMemcpyDeviceToHost));

    int errors = verify_result(tag, c_gpu, c_ref, n);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_e));
    CHECK_CUDA(cudaFree(d_c));
    return errors == 0;
  }
};

struct RawDemoCase {
  const char* tag;
  int m;
  int n;
  int k;
  bool use_pattern;
};

template <typename Demo, size_t NumCases>
int run_raw_demo_suite(const char* instruction, RawDemoCase const (&cases)[NumCases]) {
  int device = select_best_device();
  if (device < 0) {
    std::cout << "No Hopper-class GPU with available memory was found." << std::endl;
    return 0;
  }

  cudaDeviceProp props{};
  CHECK_CUDA(cudaGetDeviceProperties(&props, device));
  if (props.major < 9) {
    std::cout << "This demo requires Hopper-class hardware." << std::endl;
    return 0;
  }

  std::cout << "Raw sparse WGMMA demo: " << instruction << " on device " << device << " (" << props.name << ")"
            << std::endl;

  std::mt19937 gen(1234);
  bool ok = true;
  for (auto const& test_case : cases) {
    ok &= Demo::run_case(test_case.tag, test_case.m, test_case.n, test_case.k, test_case.use_pattern, gen);
  }

  if (!ok) {
    std::cout << "Verification FAILED" << std::endl;
    return 1;
  }

  std::cout << "Verification PASSED" << std::endl;
  return 0;
}
