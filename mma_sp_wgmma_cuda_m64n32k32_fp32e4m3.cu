#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 32>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x32x32", 64, 32, 32, true},
      {"random_64x32x128", 64, 32, 128, false},
      {"random_128x64x256", 128, 64, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n32k32.f32.e4m3.e4m3",
      kCases);
}
