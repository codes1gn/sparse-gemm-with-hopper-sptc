#include <iostream>

#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cute/tensor.hpp>

using namespace cute;

template <class MMAAtom, class AtomLayoutMNK, class PermutationMNK, class ETensor>
CUTE_HOST_DEVICE constexpr auto thrfrg_e_probe(
    TiledMMA<MMAAtom, AtomLayoutMNK, PermutationMNK> const& mma,
    ETensor&& etensor) {
  using TMma = TiledMMA<MMAAtom, AtomLayoutMNK, PermutationMNK>;

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
CUTE_HOST_DEVICE constexpr auto get_layout_e_tv_probe(TiledMMA<MArgs...> const& mma) {
  auto ref_e = make_layout(make_shape(tile_size<0>(mma), tile_size<2>(mma)));
  auto layout_e_tv = thrfrg_e_probe(mma, ref_e);

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
CUTE_HOST_DEVICE constexpr auto partition_e_probe(ThrMMA<MArgs...> const& thr_mma, ETensor&& etensor) {
  auto thr_tensor = make_tensor(static_cast<ETensor&&>(etensor).data(), thrfrg_e_probe(thr_mma, etensor.layout()));
  auto thr_vmk = make_coord(
      get<0>(thr_mma.thr_vmnk_),
      make_coord(get<1>(thr_mma.thr_vmnk_), get<3>(thr_mma.thr_vmnk_)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

int main() {
  using MmaOp = SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_SS_TN<>;
  using TiledMma = decltype(make_tiled_mma(MmaOp{}));
  using ElementEMma = sparse_elem<8, uint8_t>;
  using TensorEAtom = decltype(make_ordered_layout(
      Shape<Shape<_8, _2, _4>, Shape<_32, _2, Int<1>>>{},
      Step<Step<_3, _1, _6>, Step<_0, _5, _2>>{}));
  using SmemLayoutAtomE = ComposedLayout<
      Swizzle<0, 4, 3>,
      smem_sparse_ptr_flag_bits<ElementEMma::sparsity, sizeof_bits_v<uint8_t>>,
      TensorEAtom>;
  using SmemLayoutE = decltype(tile_to_shape(SmemLayoutAtomE{}, Shape<_64, _64>{}));

  Tensor sE = make_tensor(make_smem_ptr(recast_ptr<ElementEMma>(nullptr)), SmemLayoutE{});
  Tensor sEraw = recast<uint8_t>(sE);

  TiledMma tiled_mma;
  auto copy_atom_e = Copy_Atom<AutoVectorizingCopy, uint32_t>{};
  auto smem_tiled_copy_e = make_tiled_copy_e(copy_atom_e, tiled_mma);

  for (int tid = 0; tid < 128; ++tid) {
    auto thread_mma = tiled_mma.get_thread_slice(tid);
    auto tCsE = partition_e_probe(thread_mma, sE);
    auto smem_thr_copy_e = smem_tiled_copy_e.get_thread_slice(tid);
    auto tEsE = smem_thr_copy_e.partition_S(sE);

    std::cout << "tid " << tid;
    std::cout << " partE:";
    for (int i = 0; i < size(tCsE); ++i) {
      auto coord = tCsE.layout().get_hier_coord(i);
      (void)coord;
      std::cout << ' ' << tCsE[i];
    }
    std::cout << " copyS:";
    for (int i = 0; i < size(tEsE); ++i) {
      std::cout << ' ' << tEsE[i];
    }
    std::cout << '\n';
  }
}
