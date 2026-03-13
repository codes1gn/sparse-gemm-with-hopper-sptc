#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_bfloat16, 256>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x256x32", 64, 256, 32, true},
      {"random_64x256x128", 64, 256, 128, false},
      {"random_128x512x256", 128, 512, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n256k32.f32.bf16.bf16",
      kCases);
}
