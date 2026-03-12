#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 64>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x64x32", 64, 64, 32, true},
      {"random_64x64x128", 64, 64, 128, false},
      {"random_128x128x256", 128, 128, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n64k32.f32.e4m3.e4m3",
      kCases);
}
