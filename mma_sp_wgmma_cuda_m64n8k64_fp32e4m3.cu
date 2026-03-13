#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 8>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x8x64", 64, 8, 64, true},
      {"random_64x8x256", 64, 8, 256, false},
      {"random_128x16x256", 128, 16, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n8k64.f32.e4m3.e4m3",
      kCases);
}
