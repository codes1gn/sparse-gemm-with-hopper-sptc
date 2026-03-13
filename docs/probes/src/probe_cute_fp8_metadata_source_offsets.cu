#include <cuda_runtime.h>

#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cute/tensor.hpp>

#include "wgmma_sp_demo_common.hpp"

using namespace cute;

using ElementE = sparse_elem<8, uint8_t>;
using MmaOp = SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_SS_TN<>;
using TiledMma = decltype(make_tiled_mma(MmaOp{}));
using TensorEAtom = decltype(make_ordered_layout(
    Shape<Shape<_8, _2, _4>, Shape<_32, _2, Int<1>>>{},
    Step<Step<_3, _1, _6>, Step<_0, _5, _2>>{}));
using SmemLayoutAtomE = ComposedLayout<
    Swizzle<0, 4, 3>,
    smem_sparse_ptr_flag_bits<ElementE::sparsity, sizeof_bits_v<uint8_t>>,
    TensorEAtom>;
using SmemLayoutE = decltype(tile_to_shape(SmemLayoutAtomE{}, Shape<_64, _64>{}));

struct SharedStorage {
  alignas(128) uint8_t smem_E[cosize_v<SmemLayoutE>];
};

__global__ void probe_offsets(int* offsets, int* counts) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  __shared__ SharedStorage shared;
  int tid = threadIdx.x;

  Tensor sE = make_tensor(make_smem_ptr(recast_ptr<ElementE>(shared.smem_E)), SmemLayoutE{});
  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(tid);

  auto copy_atom_e = Copy_Atom<AutoVectorizingCopy, uint32_t>{};
  auto smem_tiled_copy_e = make_tiled_copy_e(copy_atom_e, tiled_mma);
  auto smem_thr_copy_e = smem_tiled_copy_e.get_thread_slice(tid);
  Tensor tEsE = smem_thr_copy_e.partition_S(sE);

  int count = size(tEsE);
  counts[tid] = count;
  for (int i = 0; i < count; ++i) {
    auto* ptr = reinterpret_cast<uint8_t*>(&tEsE(i));
    offsets[tid * 8 + i] = static_cast<int>(ptr - shared.smem_E);
  }
  for (int i = count; i < 8; ++i) {
    offsets[tid * 8 + i] = -1;
  }
#endif
}

int main() {
  int device = select_best_device();
  if (device < 0) return 0;

  int* d_offsets = nullptr;
  int* d_counts = nullptr;
  CHECK_CUDA(cudaMalloc(&d_offsets, sizeof(int) * 128 * 8));
  CHECK_CUDA(cudaMalloc(&d_counts, sizeof(int) * 128));
  CHECK_CUDA(cudaMemset(d_offsets, 0xff, sizeof(int) * 128 * 8));
  CHECK_CUDA(cudaMemset(d_counts, 0, sizeof(int) * 128));

  probe_offsets<<<1, 128>>>(d_offsets, d_counts);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int> offsets(128 * 8);
  std::vector<int> counts(128);
  CHECK_CUDA(cudaMemcpy(offsets.data(), d_offsets, sizeof(int) * offsets.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(counts.data(), d_counts, sizeof(int) * counts.size(), cudaMemcpyDeviceToHost));

  for (int tid = 0; tid < 128; ++tid) {
    std::cout << "tid " << tid << " count=" << counts[tid] << ':';
    for (int i = 0; i < counts[tid]; ++i) {
      std::cout << ' ' << offsets[tid * 8 + i];
    }
    std::cout << '\n';
  }

  CHECK_CUDA(cudaFree(d_offsets));
  CHECK_CUDA(cudaFree(d_counts));
  return 0;
}
