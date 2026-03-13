#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 128>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x128x64", 64, 128, 64, true},
      {"random_64x128x256", 64, 128, 256, false},
      {"random_128x128x256", 128, 128, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n128k64.f32.e4m3.e4m3",
      kCases);
}
