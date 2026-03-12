#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cute/tensor.hpp>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status__ = (func);                                             \
    if (status__ != cudaSuccess) {                                             \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status__)             \
                << " at line " << __LINE__ << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

using namespace cute;

template <class MMAAtom, class AtomLayoutMNK, class PermutationMNK, class ETensor>
CUTE_HOST_DEVICE constexpr auto thrfrg_e(
    TiledMMA<MMAAtom, AtomLayoutMNK, PermutationMNK> const& mma,
    ETensor&& etensor) {
  using TMma = TiledMMA<MMAAtom, AtomLayoutMNK, PermutationMNK>;

  CUTE_STATIC_ASSERT_V(rank(etensor) >= Int<2>{});

  auto t_tile = make_tile(get<0>(PermutationMNK{}), get<2>(PermutationMNK{}));
  auto t_tensor = logical_divide(etensor, t_tile);

  auto e_tile = make_tile(
      make_layout(size<0>(typename TMma::AtomShape_MNK{})),
      make_layout(size<2>(typename TMma::AtomShape_MNK{})));
  auto e_tensor = zipped_divide(t_tensor, e_tile);

  using AtomLayoutE_TV = typename TMma::Atom::Traits::ELayout;
  auto tv_tensor = e_tensor.compose(AtomLayoutE_TV{}, _);

  auto thr_tile = make_tile(
      _,
      make_tile(
          make_layout(size<1>(mma.thr_layout_vmnk_)),
          make_layout(size<3>(mma.thr_layout_vmnk_))));
  return zipped_divide(tv_tensor, thr_tile);
}

template <class... MArgs>
CUTE_HOST_DEVICE constexpr auto get_layout_e_tv(TiledMMA<MArgs...> const& mma) {
  auto ref_e = make_layout(make_shape(tile_size<0>(mma), tile_size<2>(mma)));
  auto layout_e_tv = thrfrg_e(mma, ref_e);

  auto etile = make_tile(
      _,
      make_tile(
          make_layout(
              make_shape(size<1>(mma.thr_layout_vmnk_), size<2>(mma.thr_layout_vmnk_)),
              make_stride(Int<1>{}, Int<0>{})),
          _));

  auto thridx_to_thrid = right_inverse(mma.thr_layout_vmnk_);
  return layout_e_tv.compose(etile, _).compose(thridx_to_thrid, _);
}

template <class... MArgs, class ETensor>
CUTE_HOST_DEVICE constexpr auto partition_e(ThrMMA<MArgs...> const& thr_mma, ETensor&& etensor) {
  auto thr_tensor = make_tensor(static_cast<ETensor&&>(etensor).data(), thrfrg_e(thr_mma, etensor.layout()));
  auto thr_vmk = make_coord(
      get<0>(thr_mma.thr_vmnk_),
      make_coord(get<1>(thr_mma.thr_vmnk_), get<3>(thr_mma.thr_vmnk_)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

template <class... CArgs, class... MArgs>
CUTE_HOST_DEVICE constexpr auto make_tiled_copy_e(
    Copy_Atom<CArgs...> const& copy_atom,
    TiledMMA<MArgs...> const& mma) {
  return make_tiled_copy_impl(copy_atom, get_layout_e_tv(mma), make_shape(tile_size<0>(mma), tile_size<2>(mma)));
}

template <typename T>
inline float elem_to_float(T x) {
  return static_cast<float>(x);
}

template <typename T>
inline T float_to_elem(float x) {
  return T{x};
}

template <typename Element>
void init_structured_sparse_a(
    std::vector<Element>& a_dense,
    int m,
    int k,
    std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> pick(0, 3);

  a_dense.assign(m * k, Element{0.0f});
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
  a_dense.assign(m * k, Element{0.0f});
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
  a_sparse.assign(m * (k / 2), Element{0.0f});
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

      uint8_t nibble = static_cast<uint8_t>(idxs[0] | (idxs[1] << 2));
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

template <typename Demo>
__global__ void wgmma_sparse_demo_kernel(
    typename Demo::ElementA const* __restrict__ a_sparse,
    typename Demo::ElementB const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    typename Demo::ElementC* __restrict__ c,
    int n,
    int k) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using ElementA = typename Demo::ElementA;
  using ElementAMma = typename Demo::ElementAMma;
  using ElementB = typename Demo::ElementB;
  using ElementC = typename Demo::ElementC;
  using ElementEMma = typename Demo::ElementEMma;
  using SharedStorage = typename Demo::SharedStorage;
  using SmemLayoutA = typename Demo::SmemLayoutA;
  using SmemLayoutB = typename Demo::SmemLayoutB;
  using SmemLayoutE = typename Demo::SmemLayoutE;
  using TiledMma = typename Demo::TiledMma;

  __shared__ SharedStorage shared;

  int tid = threadIdx.x;
  int block_row = blockIdx.y * Demo::kBlockM;
  int block_col = blockIdx.x * Demo::kBlockN;

  Tensor sA = make_tensor(make_smem_ptr(recast_ptr<ElementAMma>(shared.smem_A)), SmemLayoutA{});
  Tensor sAraw = recast<ElementA>(sA);
  Tensor sB = make_tensor(make_smem_ptr(recast_ptr<ElementB>(shared.smem_B)), SmemLayoutB{});
  Tensor sE = make_tensor(make_smem_ptr(recast_ptr<ElementEMma>(shared.smem_E)), SmemLayoutE{});
  Tensor sEraw = recast<uint8_t>(sE);

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(tid);

  Tensor gC = make_tensor(
      make_gmem_ptr(c + block_row * n + block_col),
      make_shape(Int<Demo::kBlockM>{}, Int<Demo::kBlockN>{}),
      make_stride(n, Int<1>{}));
  Tensor tCgC = thread_mma.partition_C(gC);
  Tensor tCrC = thread_mma.make_fragment_C(tCgC);
  clear(tCrC);

  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);

  Tensor tCsE = partition_e(thread_mma, sE);
  Tensor tCrE = make_fragment_like<ElementEMma>(tCsE);

  auto copy_atom_e = Copy_Atom<AutoVectorizingCopy, uint32_t>{};
  auto smem_tiled_copy_e = make_tiled_copy_e(copy_atom_e, tiled_mma);
  auto smem_thr_copy_e = smem_tiled_copy_e.get_thread_slice(tid);
  Tensor tEsE = smem_thr_copy_e.partition_S(sE);
  Tensor tErE = smem_thr_copy_e.retile_D(tCrE);

  bool first_k_tile = true;
  for (int k_tile = 0; k_tile < k; k_tile += Demo::kBlockK) {
    for (int idx = tid; idx < Demo::kBlockM * Demo::kSparseK; idx += blockDim.x) {
      int row = idx / Demo::kSparseK;
      int col = idx % Demo::kSparseK;
      sAraw(row, col) = a_sparse[(block_row + row) * (k / 2) + (k_tile / 2) + col];
    }

    for (int idx = tid; idx < Demo::kBlockN * Demo::kBlockK; idx += blockDim.x) {
      int col = idx / Demo::kBlockK;
      int kk = idx % Demo::kBlockK;
      sB(col, kk) = b[(k_tile + kk) * n + (block_col + col)];
    }

    for (int idx = tid; idx < Demo::kBlockM * Demo::kMetaBytes; idx += blockDim.x) {
      int row = idx / Demo::kMetaBytes;
      int byte_col = idx % Demo::kMetaBytes;
      sEraw(row, byte_col) = e_bytes[(block_row + row) * (k / 8) + (k_tile / 8) + byte_col];
    }

    __syncthreads();

    copy(smem_tiled_copy_e, tEsE, tErE);

    tiled_mma.accumulate_ = first_k_tile ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    CUTE_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      gemm(tiled_mma, make_zip_tensor(tCrA(_, _, k_block), tErE(_, _, k_block)), tCrB(_, _, k_block), tCrC);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    __syncthreads();
    first_k_tile = false;
  }

  axpby(1.0f, tCrC, 0.0f, tCgC);
#endif
}

template <typename ElementA_, typename ElementB_, typename MmaOp_, int BlockN_>
struct WgmmaSparseDemo {
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = float;
  using ElementAMma = sparse_elem<2, ElementA>;
  using ElementEMma = sparse_elem<8, uint8_t>;

  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = BlockN_;
  static constexpr int kBlockK = 32;
  static constexpr int kSparseK = kBlockK / 2;
  static constexpr int kMetaBytes = kBlockK / 8;

  using MmaOp = MmaOp_;
  using TiledMma = decltype(make_tiled_mma(MmaOp{}));

  using SmemLayoutA = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_SpAtom<ElementA, 2>{},
      Shape<_64, _32>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SM90::GMMA::Layout_MN_INTER_Atom<ElementB>{},
      Shape<Int<kBlockN>, _32>{}));
  using TensorEAtom = decltype(make_ordered_layout(
      Shape<Shape<_8, _2, _4>, Shape<_16, _2, Int<1>>>{},
      Step<Step<_3, _1, _5>, Step<_0, _4, _2>>{}));
  using SmemLayoutAtomE = ComposedLayout<
      Swizzle<0, 4, 3>,
      smem_sparse_ptr_flag_bits<ElementEMma::sparsity, sizeof_bits_v<uint8_t>>,
      TensorEAtom>;
  using SmemLayoutE = decltype(tile_to_shape(SmemLayoutAtomE{}, Shape<_64, _32>{}));

  struct SharedStorage {
    alignas(128) uint16_t smem_A[cosize_v<SmemLayoutA>];
    alignas(128) uint16_t smem_B[cosize_v<SmemLayoutB>];
    alignas(128) uint8_t smem_E[cosize_v<SmemLayoutE>];
  };

  static_assert(size(TiledMma{}) == Int<128>{}, "Sparse WGMMA expects one warpgroup.");

  static bool run_case(
      const std::string& tag,
      int m,
      int n,
      int k,
      bool use_pattern,
      std::mt19937& gen) {
    if ((m % kBlockM) != 0 || (n % kBlockN) != 0 || (k % kBlockK) != 0) {
      std::cerr << tag << " requires M%64==0, N%" << kBlockN << "==0, K%32==0" << std::endl;
      return false;
    }

    std::vector<ElementA> a_dense;
    std::vector<ElementB> b;
    if (use_pattern) {
      init_pattern_sparse_a(a_dense, m, k);
      init_pattern_b(b, k, n);
    } else {
      init_structured_sparse_a(a_dense, m, k, gen);
      init_random_b(b, k, n, gen);
    }

    std::vector<ElementA> a_sparse;
    std::vector<uint8_t> e_bytes;
    compress_structured_sparse_a(a_dense, a_sparse, e_bytes, m, k);

    std::vector<float> c_ref;
    cpu_gemm_ref(a_dense, b, c_ref, m, n, k);

    ElementA* d_a = nullptr;
    ElementB* d_b = nullptr;
    uint8_t* d_e = nullptr;
    ElementC* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, sizeof(ElementA) * a_sparse.size()));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(ElementB) * b.size()));
    CHECK_CUDA(cudaMalloc(&d_e, sizeof(uint8_t) * e_bytes.size()));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(ElementC) * c_ref.size()));

    CHECK_CUDA(cudaMemcpy(d_a, a_sparse.data(), sizeof(ElementA) * a_sparse.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), sizeof(ElementB) * b.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_e, e_bytes.data(), sizeof(uint8_t) * e_bytes.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_c, 0, sizeof(ElementC) * c_ref.size()));

    dim3 block(size(TiledMma{}));
    dim3 grid(n / kBlockN, m / kBlockM);
    wgmma_sparse_demo_kernel<WgmmaSparseDemo><<<grid, block>>>(d_a, d_b, d_e, d_c, n, k);
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

struct DemoCase {
  const char* tag;
  int m;
  int n;
  int k;
  bool use_pattern;
};

template <typename Demo, size_t NumCases>
int run_demo_suite(const char* instruction, DemoCase const (&cases)[NumCases]) {
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

  std::cout << "Sparse WGMMA demo: " << instruction << " on device " << device << " (" << props.name << ")"
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
