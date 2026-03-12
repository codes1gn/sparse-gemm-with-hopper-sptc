#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_bfloat16, 128>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x128x32", 64, 128, 32, true},
      {"random_64x128x128", 64, 128, 128, false},
      {"random_128x256x256", 128, 256, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.bf16.bf16",
      kCases);
}
