#include "wgmma_sp_raw_common.hpp"

#include <iomanip>
#include <vector>

template <int BlockN>
__global__ void probe_regs(uint32_t* regs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __shared__ uint8_t smem[RawSparseConfig<__nv_fp8_e4m3>::kESmemBytes];
  auto sE = raw_fp8_smem_e_tensor(smem);
  auto sEraw = cute::recast<uint8_t>(sE);
  int tid = threadIdx.x;
  for (int idx = tid; idx < 64 * 8; idx += blockDim.x) {
    int row = idx / 8;
    int byte_col = idx % 8;
    sEraw(row, byte_col) = static_cast<uint8_t>(idx & 0xFF);
  }
  __syncthreads();
  if (tid < 128) {
    regs[tid] = raw_fp8_metadata_u32<BlockN>(smem, tid);
  }
#endif
}

template <int BlockN>
void run_probe(const char* tag) {
  uint32_t* d_regs = nullptr;
  CHECK_CUDA(cudaMalloc(&d_regs, sizeof(uint32_t) * 128));
  CHECK_CUDA(cudaMemset(d_regs, 0, sizeof(uint32_t) * 128));
  probe_regs<BlockN><<<1, 128>>>(d_regs);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<uint32_t> regs(128);
  CHECK_CUDA(cudaMemcpy(regs.data(), d_regs, sizeof(uint32_t) * regs.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_regs));

  std::cout << tag << '\n';
  for (int tid = 0; tid < 128; ++tid) {
    uint32_t reg = regs[tid];
    std::cout << "tid " << std::setw(3) << tid << ": 0x"
              << std::hex << std::setw(8) << std::setfill('0') << reg
              << std::dec << std::setfill(' ') << " bytes"
              << ' ' << static_cast<int>(reg & 0xFF)
              << ' ' << static_cast<int>((reg >> 8) & 0xFF)
              << ' ' << static_cast<int>((reg >> 16) & 0xFF)
              << ' ' << static_cast<int>((reg >> 24) & 0xFF)
              << '\n';
  }
}

int main() {
  int device = select_best_device();
  if (device < 0) {
    return 0;
  }
  run_probe<8>("BlockN=8");
  run_probe<16>("BlockN=16");
  run_probe<32>("BlockN=32");
  run_probe<64>("BlockN=64");
  run_probe<128>("BlockN=128");
  run_probe<256>("BlockN=256");
  return 0;
}
