#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 16>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x16x64", 64, 16, 64, true},
      {"random_64x16x256", 64, 16, 256, false},
      {"random_128x32x256", 128, 32, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n16k64.f32.e4m3.e4m3",
      kCases);
}
