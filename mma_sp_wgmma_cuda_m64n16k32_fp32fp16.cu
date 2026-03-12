#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<half, 16>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x16x32", 64, 16, 32, true},
      {"random_64x16x128", 64, 16, 128, false},
      {"random_128x64x256", 128, 64, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n16k32.f32.f16.f16",
      kCases);
}
