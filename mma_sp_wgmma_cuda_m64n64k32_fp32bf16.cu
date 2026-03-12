#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_bfloat16, 64>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x64x32", 64, 64, 32, true},
      {"random_64x64x128", 64, 64, 128, false},
      {"random_128x128x256", 128, 128, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n64k32.f32.bf16.bf16",
      kCases);
}
