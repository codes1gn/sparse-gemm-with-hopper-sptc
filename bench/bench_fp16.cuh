#pragma once

#include <cuda_runtime.h>
#include "bench_common.hpp"

__global__ void wgmma_sp_raw_kernel_fp16_n256(
    half const* __restrict__ a_sparse,
    half const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    float* __restrict__ c,
    int n,
    int k);

__global__ void wgmma_sp_raw_kernel_fp16_n128(
    half const* __restrict__ a_sparse,
    half const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    float* __restrict__ c,
    int n,
    int k);

constexpr int kBlockM = 64;
constexpr int kThreads = 128;
constexpr int kBlockK_fp16 = 32;
constexpr int kSparseK_fp16 = kBlockK_fp16 / 2;
constexpr int kMetaBytes_fp16 = kBlockK_fp16 / 8;

template <int BlockN>
struct SparseConfig_fp16 {
  static constexpr int kASmemElements = kBlockM * kSparseK_fp16;
  static constexpr int kESmemBytes = kBlockM * kMetaBytes_fp16;
  static constexpr uint32_t kADescLeading = 64;
  static constexpr uint32_t kADescStride = 8;
};

template <int BlockN>
struct SharedStorage_fp16 {
  alignas(128) half smem_A[kBlockM * kSparseK_fp16];
  alignas(128) half smem_B[BlockN * kBlockK_fp16];
  alignas(128) uint8_t smem_E[kBlockM * kMetaBytes_fp16];
};

__device__ inline uint32_t smem_ptr_as_uint(void const* ptr) {
  uint64_t addr64;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr64) : "l"(ptr));
  return static_cast<uint32_t>(addr64);
}

__device__ inline uint64_t make_gmma_desc(uint32_t smem_addr, uint32_t leading, uint32_t stride) {
  uint64_t desc = static_cast<uint64_t>((smem_addr >> 4) & 0x3fff);
  desc |= static_cast<uint64_t>(leading) << 16;
  desc |= static_cast<uint64_t>(stride) << 32;
  return desc;
}

__device__ inline void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ inline void warpgroup_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ inline void warpgroup_commit_batch() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int NumRegs>
__device__ inline void warpgroup_fence_accum(float (&regs)[NumRegs]) {
#pragma unroll
  for (int i = 0; i < NumRegs; ++i) {
    asm volatile("" : "+f"(regs[i]) :: "memory");
  }
}

__device__ inline uint32_t ld_shared_u32(void const* ptr) {
  uint32_t value = 0;
  uint32_t addr = smem_ptr_as_uint(ptr);
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(addr));
  return value;
}

__device__ inline int smem_a_index_fp16(int row, int col) {
  return (col & 7) + ((row >> 3) * 64) + ((row & 7) * 8) + ((col >> 3) * 512);
}

template <int BlockN>
__device__ inline int smem_b_index_fp16(int col, int kk) {
  return (col & 7) + ((col >> 3) * 64) + ((kk >> 3) * (BlockN * 8)) + ((kk & 7) * 8);
}

__device__ inline int smem_e_index_fp16(int row, int byte_col) {
  return (byte_col & 1) + ((byte_col >> 1) * 32) + ((row >> 4) * 64) + ((row & 7) * 4) + (((row >> 3) & 1) * 2);
}

__device__ inline int e_thread_byte_offset_fp16(int tid) {
  return ((tid & 1) * 32) + ((tid >> 5) * 64) + (((tid >> 2) & 7) * 4);
}

template <int BlockN>
struct BDescParams_fp16 {
  static constexpr uint32_t kLeading = BlockN;
  static constexpr uint32_t kStride = BlockN == 8 ? 0u : 8u;
};

__device__ inline void wgmma_sp_fp16_n256_fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %132, 0;\n"
      "  wgmma.mma_async.sp.sync.aligned.m64n256k32.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, "
      "%72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, "
      "%88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, "
      "%104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, %130, %131, p, %133, %134, %135, %136;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
}

__device__ inline void wgmma_sp_fp16_n128_fma(uint64_t desc_a, uint64_t desc_b, float* d, uint32_t e, int scale_d) {
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %68, 0;\n"
      "  wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      "%8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, "
      "%40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, "
      "%56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, %66, %67, p, %69, %70, %71, %72;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(e), "n"(0), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(1));
}

template <int BlockN>
__device__ inline void store_accum_fp16(float const* accum, float* c, int n, int block_row, int block_col, int tid) {
  int row = block_row + ((tid >> 5) * 16) + ((tid >> 2) & 7);
  int col = block_col + ((tid & 3) * 2);
#pragma unroll
  for (int group = 0; group < BlockN / 8; ++group) {
    int idx = group * 4;
    int col_group = col + group * 8;
    c[row * n + col_group + 0] = accum[idx + 0];
    c[row * n + col_group + 1] = accum[idx + 1];
    c[(row + 8) * n + col_group + 0] = accum[idx + 2];
    c[(row + 8) * n + col_group + 1] = accum[idx + 3];
  }
}

__global__ void wgmma_sp_raw_kernel_fp16_n256(
    half const* __restrict__ a_sparse,
    half const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    float* __restrict__ c,
    int n,
    int k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __shared__ SharedStorage_fp16<256> shared;

  constexpr int kAccRegs = 128;
  int tid = threadIdx.x;
  int block_row = blockIdx.y * kBlockM;
  int block_col = blockIdx.x * 256;

  uint64_t desc_a = make_gmma_desc(
      smem_ptr_as_uint(shared.smem_A),
      SparseConfig_fp16<256>::kADescLeading,
      SparseConfig_fp16<256>::kADescStride);
  uint64_t desc_b = make_gmma_desc(
      smem_ptr_as_uint(shared.smem_B),
      256u,
      8u);

  float accum[kAccRegs];
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    accum[i] = 0.0f;
  }

  bool first_k_tile = true;
  for (int k_tile = 0; k_tile < k; k_tile += kBlockK_fp16) {
    for (int idx = tid; idx < kBlockM * kSparseK_fp16; idx += kThreads) {
      int row = idx / kSparseK_fp16;
      int col = idx % kSparseK_fp16;
      shared.smem_A[smem_a_index_fp16(row, col)] =
          a_sparse[(block_row + row) * (k / 2) + (k_tile / 2) + col];
    }

    for (int idx = tid; idx < 256 * kBlockK_fp16; idx += kThreads) {
      int col = idx / kBlockK_fp16;
      int kk = idx % kBlockK_fp16;
      shared.smem_B[smem_b_index_fp16<256>(col, kk)] =
          b[(k_tile + kk) * n + (block_col + col)];
    }

    for (int idx = tid; idx < kBlockM * kMetaBytes_fp16; idx += kThreads) {
      int row = idx / kMetaBytes_fp16;
      int byte_col = idx % kMetaBytes_fp16;
      shared.smem_E[smem_e_index_fp16(row, byte_col)] =
          e_bytes[(block_row + row) * (k / 8) + (k_tile / 8) + byte_col];
    }

    __syncthreads();

    uint32_t e = ld_shared_u32(shared.smem_E + e_thread_byte_offset_fp16(tid));
    int scale_d = first_k_tile ? 0 : 1;

    warpgroup_fence_accum(accum);
    warpgroup_arrive();
    wgmma_sp_fp16_n256_fma(desc_a, desc_b, accum, e, scale_d);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_accum(accum);

    __syncthreads();
    first_k_tile = false;
  }

  store_accum_fp16<256>(accum, c, n, block_row, block_col, tid);
#endif
}

__global__ void wgmma_sp_raw_kernel_fp16_n128(
    half const* __restrict__ a_sparse,
    half const* __restrict__ b,
    uint8_t const* __restrict__ e_bytes,
    float* __restrict__ c,
    int n,
    int k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  __shared__ SharedStorage_fp16<128> shared;

  constexpr int kAccRegs = 64;
  int tid = threadIdx.x;
  int block_row = blockIdx.y * kBlockM;
  int block_col = blockIdx.x * 128;

  uint64_t desc_a = make_gmma_desc(
      smem_ptr_as_uint(shared.smem_A),
      SparseConfig_fp16<128>::kADescLeading,
      SparseConfig_fp16<128>::kADescStride);
  uint64_t desc_b = make_gmma_desc(
      smem_ptr_as_uint(shared.smem_B),
      128u,
      8u);

  float accum[kAccRegs];
#pragma unroll
  for (int i = 0; i < kAccRegs; ++i) {
    accum[i] = 0.0f;
  }

  bool first_k_tile = true;
  for (int k_tile = 0; k_tile < k; k_tile += kBlockK_fp16) {
    for (int idx = tid; idx < kBlockM * kSparseK_fp16; idx += kThreads) {
      int row = idx / kSparseK_fp16;
      int col = idx % kSparseK_fp16;
      shared.smem_A[smem_a_index_fp16(row, col)] =
          a_sparse[(block_row + row) * (k / 2) + (k_tile / 2) + col];
    }

    for (int idx = tid; idx < 128 * kBlockK_fp16; idx += kThreads) {
      int col = idx / kBlockK_fp16;
      int kk = idx % kBlockK_fp16;
      shared.smem_B[smem_b_index_fp16<128>(col, kk)] =
          b[(k_tile + kk) * n + (block_col + col)];
    }

    for (int idx = tid; idx < kBlockM * kMetaBytes_fp16; idx += kThreads) {
      int row = idx / kMetaBytes_fp16;
      int byte_col = idx % kMetaBytes_fp16;
      shared.smem_E[smem_e_index_fp16(row, byte_col)] =
          e_bytes[(block_row + row) * (k / 8) + (k_tile / 8) + byte_col];
    }

    __syncthreads();

    uint32_t e = ld_shared_u32(shared.smem_E + e_thread_byte_offset_fp16(tid));
    int scale_d = first_k_tile ? 0 : 1;

    warpgroup_fence_accum(accum);
    warpgroup_arrive();
    wgmma_sp_fp16_n128_fma(desc_a, desc_b, accum, e, scale_d);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_accum(accum);

    __syncthreads();
    first_k_tile = false;
  }

  store_accum_fp16<128>(accum, c, n, block_row, block_col, tid);
#endif
}

bool run_bench_fp16_n256(
    const std::string& tag,
    int m,
    int n,
    int k,
    std::mt19937& gen) {
  if ((m % kBlockM) != 0 || (n % 256) != 0 || (k % kBlockK_fp16) != 0) {
    std::cerr << tag << " requires M%64==0, N%256==0, K%32==0" << std::endl;
    return false;
  }

  std::vector<half> a_dense;
  std::vector<half> b;
  init_structured_sparse_a(a_dense, m, k, gen);
  init_random_b(b, k, n, gen);

  std::vector<half> a_sparse;
  std::vector<uint8_t> e_bytes;
  compress_structured_sparse_a(a_dense, a_sparse, e_bytes, m, k);

  std::vector<float> c_ref;
  cpu_gemm_ref(a_dense, b, c_ref, m, n, k);

  half* d_a = nullptr;
  half* d_b = nullptr;
  uint8_t* d_e = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(half) * a_sparse.size()));
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(half) * b.size()));
  CHECK_CUDA(cudaMalloc(&d_e, sizeof(uint8_t) * e_bytes.size()));
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * c_ref.size()));

  CHECK_CUDA(cudaMemcpy(d_a, a_sparse.data(), sizeof(half) * a_sparse.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), sizeof(half) * b.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_e, e_bytes.data(), sizeof(uint8_t) * e_bytes.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float) * c_ref.size()));

  dim3 block(kThreads);
  dim3 grid(n / 256, m / kBlockM);
  
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  
  constexpr int kNumIter = 10;
  
  for (int iter = 0; iter < 3; iter++) {
    wgmma_sp_raw_kernel_fp16_n256<<<grid, block>>>(d_a, d_b, d_e, d_c, n, k);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  
  CHECK_CUDA(cudaEventRecord(start));
  for (int iter = 0; iter < kNumIter; iter++) {
    wgmma_sp_raw_kernel_fp16_n256<<<grid, block>>>(d_a, d_b, d_e, d_c, n, k);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  
  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  
  double tflops = (2.0 * m * n * k * kNumIter) / (milliseconds * 1e-3) / 1e12;
  std::cout << tag << ": " << milliseconds / kNumIter << " ms avg, " << tflops << " TFLOPS" << std::endl;
  
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  std::vector<float> c_gpu(c_ref.size(), 0.0f);
  CHECK_CUDA(cudaMemcpy(c_gpu.data(), d_c, sizeof(float) * c_gpu.size(), cudaMemcpyDeviceToHost));

  int errors = verify_result(tag, c_gpu, c_ref, n);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_e));
  CHECK_CUDA(cudaFree(d_c));
  return errors == 0;
}

bool run_bench_fp16_n128(
    const std::string& tag,
    int m,
    int n,
    int k,
    std::mt19937& gen) {
  if ((m % kBlockM) != 0 || (n % 128) != 0 || (k % kBlockK_fp16) != 0) {
    std::cerr << tag << " requires M%64==0, N%128==0, K%32==0" << std::endl;
    return false;
  }

  std::vector<half> a_dense;
  std::vector<half> b;
  init_structured_sparse_a(a_dense, m, k, gen);
  init_random_b(b, k, n, gen);

  std::vector<half> a_sparse;
  std::vector<uint8_t> e_bytes;
  compress_structured_sparse_a(a_dense, a_sparse, e_bytes, m, k);

  std::vector<float> c_ref;
  cpu_gemm_ref(a_dense, b, c_ref, m, n, k);

  half* d_a = nullptr;
  half* d_b = nullptr;
  uint8_t* d_e = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(half) * a_sparse.size()));
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(half) * b.size()));
  CHECK_CUDA(cudaMalloc(&d_e, sizeof(uint8_t) * e_bytes.size()));
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * c_ref.size()));

  CHECK_CUDA(cudaMemcpy(d_a, a_sparse.data(), sizeof(half) * a_sparse.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), sizeof(half) * b.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_e, e_bytes.data(), sizeof(uint8_t) * e_bytes.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_c, 0, sizeof(float) * c_ref.size()));

  dim3 block(kThreads);
  dim3 grid(n / 128, m / kBlockM);
  
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  
  constexpr int kNumIter = 10;
  
  for (int iter = 0; iter < 3; iter++) {
    wgmma_sp_raw_kernel_fp16_n128<<<grid, block>>>(d_a, d_b, d_e, d_c, n, k);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  
  CHECK_CUDA(cudaEventRecord(start));
  for (int iter = 0; iter < kNumIter; iter++) {
    wgmma_sp_raw_kernel_fp16_n128<<<grid, block>>>(d_a, d_b, d_e, d_c, n, k);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  
  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  
  double tflops = (2.0 * m * n * k * kNumIter) / (milliseconds * 1e-3) / 1e12;
  std::cout << tag << ": " << milliseconds / kNumIter << " ms avg, " << tflops << " TFLOPS" << std::endl;
  
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  std::vector<float> c_gpu(c_ref.size(), 0.0f);
  CHECK_CUDA(cudaMemcpy(c_gpu.data(), d_c, sizeof(float) * c_gpu.size(), cudaMemcpyDeviceToHost));

  int errors = verify_result(tag, c_gpu, c_ref, n);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_e));
  CHECK_CUDA(cudaFree(d_c));
  return errors == 0;
}

bool run_fp16_bench_suite(int m, int n, int k) {
  int device = select_best_device();
  if (device < 0) {
    std::cout << "No Hopper-class GPU with available memory was found." << std::endl;
    return false;
  }

  cudaDeviceProp props{};
  CHECK_CUDA(cudaGetDeviceProperties(&props, device));
  if (props.major < 9) {
    std::cout << "This demo requires Hopper-class hardware." << std::endl;
    return false;
  }

  std::cout << "FP16 sparse WGMMA benchmark: device " << device << " (" << props.name << ")" << std::endl;
  std::cout << "Matrix size: " << m << "x" << n << "x" << k << std::endl;

  std::mt19937 gen(1234);
  bool ok = true;

  ok &= run_bench_fp16_n256("fp16_n256", m, n, k, gen);
  ok &= run_bench_fp16_n128("fp16_n128", m, n, k, gen);

  if (!ok) {
    std::cout << "Verification FAILED" << std::endl;
    return false;
  }

  std::cout << "Verification PASSED" << std::endl;
  return true;
}
