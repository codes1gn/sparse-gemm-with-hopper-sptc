#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<half, 8>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x8x32", 64, 8, 32, true},
      {"random_64x8x128", 64, 8, 128, false},
      {"random_128x32x256", 128, 32, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.f16.f16",
      kCases);
}
