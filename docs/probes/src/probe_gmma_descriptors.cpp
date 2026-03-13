#include <iostream>

#include <cuda_fp8.h>

#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma_sparse.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_sparse.hpp>
#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  auto dump = [](const char* tag, auto desc) {
    std::cout << tag << " desc=0x" << std::hex << static_cast<unsigned long long>(desc.desc_) << std::dec
              << " lead=" << desc.bitfield.leading_byte_offset_
              << " stride=" << desc.bitfield.stride_byte_offset_
              << " base=" << static_cast<int>(desc.bitfield.base_offset_)
              << " layout=" << static_cast<int>(desc.bitfield.layout_type_) << "\n";
  };

  auto a64 = tile_to_shape(
      SM90::GMMA::Layout_K_INTER_SpAtom<float_e4m3_t, 2>{},
      make_shape(Int<64>{}, Int<64>{}));
  auto b8 = tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      make_shape(Int<8>{}, Int<64>{}));
  auto b16 = tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      make_shape(Int<16>{}, Int<64>{}));
  auto b32 = tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      make_shape(Int<32>{}, Int<64>{}));
  auto b64 = tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      make_shape(Int<64>{}, Int<64>{}));
  auto b128 = tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      make_shape(Int<128>{}, Int<64>{}));
  auto b256 = tile_to_shape(
      SM90::GMMA::Layout_K_INTER_Atom<float_e4m3_t>{},
      make_shape(Int<256>{}, Int<64>{}));

  auto ta64 = make_tensor(make_smem_ptr<sparse_elem<2, float_e4m3_t>>(nullptr), a64);
  auto tb8 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), b8);
  auto tb16 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), b16);
  auto tb32 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), b32);
  auto tb64 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), b64);
  auto tb128 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), b128);
  auto tb256 = make_tensor(make_smem_ptr<float_e4m3_t>(nullptr), b256);

  dump("a64", SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(ta64));
  dump("b8", SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(tb8));
  dump("b16", SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(tb16));
  dump("b32", SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(tb32));
  dump("b64", SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(tb64));
  dump("b128", SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(tb128));
  dump("b256", SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::K>(tb256));

  return 0;
}
