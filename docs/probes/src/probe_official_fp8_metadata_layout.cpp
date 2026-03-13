#include <iostream>

#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cutlass/gemm/collective/builders/sm90_sparse_config.inl>

int main() {
  using namespace cute;
  using ElementA = cutlass::float_e4m3_t;
  using ElementAMma = sparse_elem<2, ElementA>;
  using ElementEMma = sparse_elem<8, uint8_t>;
  using SparseConfig = cutlass::Sm90GemmSparseConfig<ElementAMma, GMMA::Major::K, ElementEMma, Int<64>>;
  using SmemLayoutAtomE = ComposedLayout<
      Swizzle<0, 4, 3>,
      smem_sparse_ptr_flag_bits<ElementEMma::sparsity, sizeof_bits_v<uint8_t>>,
      typename SparseConfig::TensorEAtom>;
  using SmemLayoutE = decltype(tile_to_shape(SmemLayoutAtomE{}, Shape<_64, _64>{}));

  auto sE = make_tensor(make_smem_ptr(recast_ptr<ElementEMma>(nullptr)), SmemLayoutE{});
  auto sEraw = recast<uint8_t>(sE);

  std::cout << "sizes " << size<0>(sEraw) << " x " << size<1>(sEraw) << "\n";
  for (int r = 0; r < 16; ++r) {
    std::cout << "row " << r << ':';
    for (int c = 0; c < 8; ++c) {
      std::cout << ' ' << sEraw(r, c);
    }
    std::cout << '\n';
  }
}
