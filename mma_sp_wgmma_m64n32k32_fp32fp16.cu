#include "wgmma_sp_demo_common.hpp"

using Demo = WgmmaSparseDemo<
    half_t,
    half_t,
    SM90::GMMA::SPARSE::GMMA_64x32x32_F32F16F16_SS<
        SM90::GMMA::Major::K,
        SM90::GMMA::Major::MN>,
    32>;

int main() {
  constexpr DemoCase kCases[] = {
      {"pattern_64x32x32", 64, 32, 32, true},
      {"random_64x32x128", 64, 32, 128, false},
      {"random_128x64x256", 128, 64, 256, false},
  };
  return run_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n32k32.f32.f16.f16",
      kCases);
}
