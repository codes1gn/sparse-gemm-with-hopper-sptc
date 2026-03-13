#include "wgmma_sp_raw_common.hpp"

#include <iomanip>
#include <vector>

__global__ void probe_fp8_metadata_manual(
    int* logical_to_physical,
    uint8_t* physical_bytes,
    uint32_t* thread_regs) {
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

  for (int idx = tid; idx < 64 * 8; idx += blockDim.x) {
    int row = idx / 8;
    int byte_col = idx % 8;
    uint8_t* ptr = &sEraw(row, byte_col);
    logical_to_physical[idx] = static_cast<int>(ptr - smem);
  }

  for (int idx = tid; idx < RawSparseConfig<__nv_fp8_e4m3>::kESmemBytes; idx += blockDim.x) {
    physical_bytes[idx] = smem[idx];
  }

  if (tid < 128) {
    thread_regs[tid] = raw_fp8_metadata_u32<8>(smem, tid);
  }
#endif
}

int main() {
  int device = select_best_device();
  if (device < 0) {
    return 0;
  }

  int* d_map = nullptr;
  uint8_t* d_phys = nullptr;
  uint32_t* d_regs = nullptr;
  CHECK_CUDA(cudaMalloc(&d_map, sizeof(int) * 64 * 8));
  CHECK_CUDA(cudaMalloc(&d_phys, sizeof(uint8_t) * RawSparseConfig<__nv_fp8_e4m3>::kESmemBytes));
  CHECK_CUDA(cudaMalloc(&d_regs, sizeof(uint32_t) * 128));
  CHECK_CUDA(cudaMemset(d_map, 0, sizeof(int) * 64 * 8));
  CHECK_CUDA(cudaMemset(d_phys, 0, sizeof(uint8_t) * RawSparseConfig<__nv_fp8_e4m3>::kESmemBytes));
  CHECK_CUDA(cudaMemset(d_regs, 0, sizeof(uint32_t) * 128));

  probe_fp8_metadata_manual<<<1, 128>>>(d_map, d_phys, d_regs);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int> map(64 * 8);
  std::vector<uint8_t> phys(RawSparseConfig<__nv_fp8_e4m3>::kESmemBytes);
  std::vector<uint32_t> regs(128);
  CHECK_CUDA(cudaMemcpy(map.data(), d_map, sizeof(int) * map.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(phys.data(), d_phys, sizeof(uint8_t) * phys.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(regs.data(), d_regs, sizeof(uint32_t) * regs.size(), cudaMemcpyDeviceToHost));

  std::cout << "logical_to_physical\n";
  for (int row = 0; row < 64; ++row) {
    std::cout << "row " << std::setw(2) << row << ':';
    for (int byte_col = 0; byte_col < 8; ++byte_col) {
      std::cout << ' ' << std::setw(4) << map[row * 8 + byte_col];
    }
    std::cout << '\n';
  }

  std::cout << "physical_bytes_nonzeroish\n";
  for (int idx = 0; idx < static_cast<int>(phys.size()); ++idx) {
    if (phys[idx] != 0) {
      std::cout << idx << ':' << static_cast<int>(phys[idx]) << '\n';
    }
  }

  std::cout << "thread_regs\n";
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

  CHECK_CUDA(cudaFree(d_map));
  CHECK_CUDA(cudaFree(d_phys));
  CHECK_CUDA(cudaFree(d_regs));
  return 0;
}
