#include "wgmma_sp_demo_common.hpp"

using Demo = WgmmaSparseDemo<
    bfloat16_t,
    bfloat16_t,
    SM90::GMMA::SPARSE::GMMA_64x16x32_F32BF16BF16_SS<
        SM90::GMMA::Major::K,
        SM90::GMMA::Major::MN>,
    16>;

int main() {
  constexpr DemoCase kCases[] = {
      {"pattern_64x16x32", 64, 16, 32, true},
      {"random_64x16x128", 64, 16, 128, false},
      {"random_128x64x256", 128, 64, 256, false},
  };
  return run_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n16k32.f32.bf16.bf16",
      kCases);
}
