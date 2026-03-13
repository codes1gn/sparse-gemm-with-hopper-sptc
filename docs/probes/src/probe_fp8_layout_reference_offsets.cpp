#include <iostream>

#include <cuda_fp8.h>

#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  using SmemLayoutA = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_SpAtom<float_e4m3_t, 2>{},
      Shape<_64, _64>{}));
  using SmemLayoutB8 = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      Shape<_8, _64>{}));
  using SmemLayoutB16 = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      Shape<_16, _64>{}));
  using SmemLayoutB32 = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      Shape<_32, _64>{}));
  using SmemLayoutB64 = decltype(tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      Shape<_64, _64>{}));

  auto a = recast<float_e4m3_t>(make_tensor(make_smem_ptr(recast_ptr<sparse_elem<2, float_e4m3_t>>(nullptr)), SmemLayoutA{}));
  auto b8 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), SmemLayoutB8{});
  auto b16 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), SmemLayoutB16{});
  auto b32 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), SmemLayoutB32{});
  auto b64 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), SmemLayoutB64{});

  auto dump_a = [&](int row) {
    std::cout << "A row " << row << ':';
    for (int col = 0; col < 32; ++col) {
      std::cout << ' ' << a.layout()(make_coord(row, col));
    }
    std::cout << '\n';
  };

  auto dump_b = [&](const char* tag, auto layout) {
    std::cout << tag << '\n';
    for (int row = 0; row < 8; ++row) {
      std::cout << "row " << row << ':';
      for (int kk = 0; kk < 32; ++kk) {
        std::cout << ' ' << layout(make_coord(row, kk));
      }
      std::cout << '\n';
    }
  };

  dump_a(0);
  dump_a(1);
  dump_a(8);
  dump_a(16);
  dump_b("B8", b8.layout());
  dump_b("B16", b16.layout());
  dump_b("B32", b32.layout());
  dump_b("B64", b64.layout());
  return 0;
}
