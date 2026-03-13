#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 8>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"random_64x8x64", 64, 8, 64, false},
      {"random_64x8x128", 64, 8, 128, false},
  };
  return run_raw_demo_suite<Demo>(
      "tmp fp8 single-tile probe",
      kCases);
}
