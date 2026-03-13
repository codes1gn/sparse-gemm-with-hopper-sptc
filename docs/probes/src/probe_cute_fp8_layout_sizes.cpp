#include <iostream>

#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>

int main() {
  using namespace cute;
  using SmemLayoutA = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_SpAtom<float_e4m3_t, 2>{},
      Shape<_64, _64>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      Shape<_8, _64>{}));
  using TensorEAtom = decltype(make_ordered_layout(
      Shape<Shape<_8, _2, _4>, Shape<_32, _2, Int<1>>>{},
      Step<Step<_3, _1, _6>, Step<_0, _5, _2>>{}));
  using SmemLayoutAtomE = ComposedLayout<
      Swizzle<0, 4, 3>,
      smem_sparse_ptr_flag_bits<8, sizeof_bits_v<uint8_t>>,
      TensorEAtom>;
  using SmemLayoutE = decltype(tile_to_shape(SmemLayoutAtomE{}, Shape<_64, _64>{}));
  std::cout << "cosize A=" << cosize_v<SmemLayoutA> << "\n";
  std::cout << "cosize B=" << cosize_v<SmemLayoutB> << "\n";
  std::cout << "cosize E=" << cosize_v<SmemLayoutE> << "\n";
  std::cout << "sizeof sparse_elem=" << sizeof(sparse_elem<2, float_e4m3_t>) << "\n";
  std::cout << "sizeof float_e4m3_t=" << sizeof(float_e4m3_t) << "\n";
}
