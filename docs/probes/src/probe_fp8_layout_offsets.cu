#include <iostream>

#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_sparse.hpp>
#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  using ElementA = sparse_elem<2, float_e4m3_t>;
  using ElementE = sparse_elem<8, uint8_t>;

  using SmemLayoutA = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_SpAtom<float_e4m3_t, 2>{},
      Shape<_64, _64>{}));
  using SmemLayoutB8 = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      Shape<_8, _64>{}));
  using SmemLayoutB32 = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      Shape<_32, _64>{}));

  Tensor sA = make_tensor(make_smem_ptr(recast_ptr<ElementA>(nullptr)), SmemLayoutA{});
  Tensor sAraw = recast<float_e4m3_t>(sA);

  Tensor sB8 = make_tensor(make_smem_ptr(recast_ptr<float_e4m3_t>(nullptr)), SmemLayoutB8{});
  Tensor sB32 = make_tensor(make_smem_ptr(recast_ptr<float_e4m3_t>(nullptr)), SmemLayoutB32{});

  std::cout << "A raw sizes " << size<0>(sAraw) << " x " << size<1>(sAraw) << "\n";
  for (int r = 0; r < 8; ++r) {
    std::cout << "A row " << r << ':';
    for (int c = 0; c < 32; ++c) {
      std::cout << ' ' << sAraw(r, c);
    }
    std::cout << '\n';
  }

  std::cout << "B8 sizes " << size<0>(sB8) << " x " << size<1>(sB8) << "\n";
  for (int r = 0; r < 8; ++r) {
    std::cout << "B8 row " << r << ':';
    for (int c = 0; c < 32; ++c) {
      std::cout << ' ' << sB8(r, c);
    }
    std::cout << '\n';
  }

  std::cout << "B32 sizes " << size<0>(sB32) << " x " << size<1>(sB32) << "\n";
  for (int r = 0; r < 8; ++r) {
    std::cout << "B32 row " << r << ':';
    for (int c = 0; c < 32; ++c) {
      std::cout << ' ' << sB32(r, c);
    }
    std::cout << '\n';
  }

  Tensor sE = make_tensor(make_smem_ptr(recast_ptr<ElementE>(nullptr)), SM90::GMMA::ELayout_64x64{});
  Tensor sEraw = recast<uint8_t>(sE);
  std::cout << "E raw sizes " << size<0>(sEraw) << " x " << size<1>(sEraw) << "\n";
  for (int r = 0; r < 8; ++r) {
    std::cout << "E row " << r << ':';
    for (int c = 0; c < 8; ++c) {
      std::cout << ' ' << sEraw(r, c);
    }
    std::cout << '\n';
  }

  return 0;
}
