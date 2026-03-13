#include <cuda_runtime.h>

#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cute/tensor.hpp>

#include "wgmma_sp_demo_common.hpp"

using namespace cute;

using ElementA = float_e4m3_t;
using ElementB = float_e4m3_t;
using ElementC = float;
using ElementAMma = sparse_elem<2, ElementA>;
using ElementEMma = sparse_elem<8, uint8_t>;

constexpr int kBlockM = 64;
constexpr int kBlockN = 8;
constexpr int kBlockK = 64;
constexpr int kSparseK = kBlockK / 2;
constexpr int kMetaBytes = kBlockK / 8;

using MmaOp = SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_SS_TN<>;
using TiledMma = decltype(make_tiled_mma(MmaOp{}));

using SmemLayoutA = decltype(tile_to_shape(
    SM90::GMMA::Layout_K_INTER_SpAtom<ElementA, 2>{},
    Shape<_64, _64>{}));
using SmemLayoutB = decltype(tile_to_shape(
    SM90::GMMA::Layout_K_INTER_Atom<ElementB>{},
    Shape<_8, _64>{}));
using TensorEAtom = decltype(make_ordered_layout(
    Shape<Shape<_8, _2, _4>, Shape<_32, _2, Int<1>>>{},
    Step<Step<_3, _1, _6>, Step<_0, _5, _2>>{}));
using SmemLayoutAtomE = ComposedLayout<
    Swizzle<0, 4, 3>,
    smem_sparse_ptr_flag_bits<ElementEMma::sparsity, sizeof_bits_v<uint8_t>>,
    TensorEAtom>;
using SmemLayoutE = decltype(tile_to_shape(SmemLayoutAtomE{}, Shape<_64, _64>{}));

struct SharedStorage {
  alignas(128) uint16_t smem_A[cosize_v<SmemLayoutA>];
  alignas(128) uint8_t smem_B[cosize_v<SmemLayoutB>];
  alignas(128) uint8_t smem_E[cosize_v<SmemLayoutE>];
};

static_assert(size(TiledMma{}) == Int<128>{}, "Sparse WGMMA expects one warpgroup.");

template <typename Demo>
__global__ void kernel(
    typename Demo::ElementA const* __restrict__ a_sparse,
    typename Demo::ElementB const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    typename Demo::ElementC* __restrict__ c,
    int n,
    int k) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
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

struct Demo {
  using ElementA = ::ElementA;
  using ElementB = ::ElementB;
  using ElementC = ::ElementC;
  static constexpr int kBlockM = ::kBlockM;
  static constexpr int kBlockN = ::kBlockN;
  static constexpr int kBlockK = ::kBlockK;
  static constexpr int kSparseK = ::kSparseK;
  static constexpr int kMetaBytes = ::kMetaBytes;
};

int main() {
  int device = select_best_device();
  if (device < 0) return 0;

  std::mt19937 gen(1234);
  std::vector<ElementA> a_dense;
  std::vector<ElementB> b;
  init_structured_sparse_a(a_dense, 64, 64, gen);
  init_random_b(b, 64, 8, gen);

  std::vector<ElementA> a_sparse;
  std::vector<uint8_t> e_bytes;
  compress_structured_sparse_a(a_dense, a_sparse, e_bytes, 64, 64);

  std::vector<float> c_ref;
  cpu_gemm_ref(a_dense, b, c_ref, 64, 8, 64);

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

  kernel<Demo><<<dim3(1, 1), dim3(size(TiledMma{}))>>>(d_a, d_b, d_e, d_c, 8, 64);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> c_gpu(c_ref.size(), 0.0f);
  CHECK_CUDA(cudaMemcpy(c_gpu.data(), d_c, sizeof(float) * c_gpu.size(), cudaMemcpyDeviceToHost));
  int errors = verify_result("cute_fp8_k64", c_gpu, c_ref, 8);
  std::cout << (errors == 0 ? "PASS" : "FAIL") << std::endl;

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_e));
  CHECK_CUDA(cudaFree(d_c));
  return errors == 0 ? 0 : 1;
}
