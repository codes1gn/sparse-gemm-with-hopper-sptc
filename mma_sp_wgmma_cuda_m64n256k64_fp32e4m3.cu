#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 256>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x256x64", 64, 256, 64, true},
      {"random_64x256x256", 64, 256, 256, false},
      {"random_128x256x256", 128, 256, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n256k64.f32.e4m3.e4m3",
      kCases);
}
