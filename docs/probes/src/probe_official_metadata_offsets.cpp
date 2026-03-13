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

  auto f16sE = make_tensor(make_smem_ptr(recast_ptr<F16E>(nullptr)), F16SmemE{});
  auto f16raw = recast<uint8_t>(f16sE);
  auto fp8sE = make_tensor(make_smem_ptr(recast_ptr<Fp8E>(nullptr)), Fp8SmemE{});
  auto fp8raw = recast<uint8_t>(fp8sE);

  std::cout << "F16 raw size " << size<0>(f16raw) << " x " << size<1>(f16raw) << "\n";
  for (int r = 0; r < 16; ++r) {
    std::cout << "f16 row " << r << ':';
    for (int c = 0; c < 4; ++c) {
      std::cout << ' ' << f16raw.layout()(make_coord(r, c));
    }
    std::cout << '\n';
  }

  std::cout << "FP8 raw size " << size<0>(fp8raw) << " x " << size<1>(fp8raw) << "\n";
  for (int r = 0; r < 16; ++r) {
    std::cout << "fp8 row " << r << ':';
    for (int c = 0; c < 8; ++c) {
      std::cout << ' ' << fp8raw.layout()(make_coord(r, c));
    }
    std::cout << '\n';
  }
}
