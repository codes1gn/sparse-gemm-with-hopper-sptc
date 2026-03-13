#include <iostream>

#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cutlass/gemm/collective/builders/sm90_sparse_config.inl>

int main() {
  using namespace cute;

  using F16A = sparse_elem<2, cutlass::half_t>;
  using F16E = sparse_elem<8, uint8_t>;
  using F16Cfg = cutlass::Sm90GemmSparseConfig<F16A, GMMA::Major::K, F16E, Int<32>>;
  using F16SmemAtomE = ComposedLayout<
      Swizzle<0, 4, 3>,
      smem_sparse_ptr_flag_bits<F16E::sparsity, sizeof_bits_v<uint8_t>>,
      typename F16Cfg::TensorEAtom>;
  using F16SmemE = decltype(tile_to_shape(F16SmemAtomE{}, Shape<_64, _32>{}));

  using Fp8A = sparse_elem<2, cutlass::float_e4m3_t>;
  using Fp8E = sparse_elem<8, uint8_t>;
  using Fp8Cfg = cutlass::Sm90GemmSparseConfig<Fp8A, GMMA::Major::K, Fp8E, Int<64>>;
  using Fp8SmemAtomE = ComposedLayout<
      Swizzle<0, 4, 3>,
      smem_sparse_ptr_flag_bits<Fp8E::sparsity, sizeof_bits_v<uint8_t>>,
      typename Fp8Cfg::TensorEAtom>;
  using Fp8SmemE = decltype(tile_to_shape(Fp8SmemAtomE{}, Shape<_64, _64>{}));

  std::cout << "f16 TensorEAtom shape=" << size<0>(typename F16Cfg::TensorEAtom{}) << "x" << size<1>(typename F16Cfg::TensorEAtom{}) << " cosize=" << cosize_v<typename F16Cfg::TensorEAtom> << "\n";
  std::cout << "f16 SmemLayoutE cosize=" << cosize_v<F16SmemE> << "\n";
  std::cout << "fp8 TensorEAtom shape=" << size<0>(typename Fp8Cfg::TensorEAtom{}) << "x" << size<1>(typename Fp8Cfg::TensorEAtom{}) << " cosize=" << cosize_v<typename Fp8Cfg::TensorEAtom> << "\n";
  std::cout << "fp8 SmemLayoutE cosize=" << cosize_v<Fp8SmemE> << "\n";
}
