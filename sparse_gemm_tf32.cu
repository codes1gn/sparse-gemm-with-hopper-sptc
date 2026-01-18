
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <cmath>

#define CHECK_CUDA(func) \
  { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " \
                << __LINE__ << std::endl; \
      exit(EXIT_FAILURE); \
    } \
  }

#define M 16
#define N 8
#define K 16 

// ------------------------------------------------------------------------------------------------
// Device Helper
// ------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_sp_sync_tf32(float* d, const float* a, const float* b, const float* c, const int* e) {
    const int* ai = reinterpret_cast<const int*>(a);
    const int* bi = reinterpret_cast<const int*>(b);
    // Linear register order [0,1,2,3] and Selector 0x0 as per Cutlass TF32 source
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(ai[0]), "r"(ai[1]), "r"(ai[2]), "r"(ai[3]), 
          "r"(bi[0]), "r"(bi[1]), "r"(bi[2]), "r"(bi[3]), 
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), 
          "r"(e[0])
    );
}

__global__ void sparse_gemm_kernel_tf32(
    const float* __restrict__ A, 
    const float* __restrict__ B_col,
    float* __restrict__ C,
    const uint32_t* __restrict__ E
) {
    int tid = threadIdx.x;
    float a_frag[4]; float b_frag[4]; int e_frag[1]; float c_frag[4] = {0};
    __shared__ float smem_A[16 * 8]; 
    __shared__ float smem_B[16 * 8];
    __shared__ uint32_t smem_E[8];

    for(int i=0; i<4; ++i) smem_A[tid*4+i] = A[tid*4+i];
    for(int i=0; i<4; ++i) smem_B[tid*4+i] = B_col[tid*4+i];
    if (tid < 8) smem_E[tid] = E[tid];
    __syncthreads();
    
    // 1. A Fragment Mapping (Row 0/8, 1/9...)
    int group_id = tid / 4; // 0..7
    int col_base = (tid % 4) * 2; // 0, 2, 4, 6
    a_frag[0] = smem_A[group_id * 8 + col_base + 0];
    a_frag[1] = smem_A[group_id * 8 + col_base + 1];
    a_frag[2] = smem_A[(group_id + 8) * 8 + col_base + 0];
    a_frag[3] = smem_A[(group_id + 8) * 8 + col_base + 1];

    // 2. B Fragment Mapping (Column-Major smem_B k16 x n8)
    int col_b = tid / 4; // Thread handles Col 0..7
    int row_b_base = (tid % 4) * 2; // 0, 2, 4, 6
    b_frag[0] = smem_B[col_b * 16 + row_b_base + 0];
    b_frag[1] = smem_B[col_b * 16 + row_b_base + 1];
    b_frag[2] = smem_B[col_b * 16 + row_b_base + 8];
    b_frag[3] = smem_B[col_b * 16 + row_b_base + 9];
    
    // 3. Metadata (Row i | Row i+8)
    e_frag[0] = smem_E[group_id];
    mma_sp_sync_tf32(c_frag, a_frag, b_frag, c_frag, e_frag);
    
    // 4. Store C (Same mapping as A/D)
    for (int r = 0; r < 4; ++r) {
        int row = (r < 2) ? group_id : group_id + 8;
        int col = (tid % 4) * 2 + (r % 2);
        if (row < 16 && col < 8) C[row * 8 + col] = c_frag[r];
    }
}

int main() {
    std::cout << "Running Sparse GEMM TF32 (FP32) Definitive Version..." << std::endl;
    int m=16, n=8, k=16;
    std::vector<float> h_A_dense(m * k);
    std::vector<float> h_B(k * n);
    std::mt19937 gen(42); std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(int i=0; i<m*k; ++i) h_A_dense[i] = dist(gen);
    for(int i=0; i<k*n; ++i) h_B[i] = dist(gen);
    
    // Convert B to Col-Major
    std::vector<float> h_B_col(k * n);
    for(int r=0; r<k; ++r) for(int c=0; c<n; ++c) h_B_col[c*k + r] = h_B[r*n + c];

    // Sparse A (Linear first 2 kept)
    std::vector<float> h_A_sparse(m * (k/2));
    for(int r=0; r<m; ++r) {
        for(int cg=0; cg<k/4; ++cg) {
            h_A_sparse[r*(k/2) + cg*2 + 0] = h_A_dense[r*k + cg*4 + 0];
            h_A_sparse[r*(k/2) + cg*2 + 1] = h_A_dense[r*k + cg*4 + 1];
        }
    }
    
    // Metadata: Interleaved (Row i | Row i+8)
    std::vector<uint32_t> h_E(8);
    for(int i=0; i<8; ++i) {
        uint32_t row_i = 0x4444; // indices (0,1) for 4 groups: 0x4444 is (0,1,0,1,0,1,0,1) bitpairs? 
        // 0 << 0 | 1 << 2 = 0x4. 4 groups: 0x4444.
        h_E[i] = (row_i << 16) | row_i; // Row i & Row i+8
    }

    float *d_A, *d_B, *d_C; uint32_t *d_E;
    CHECK_CUDA(cudaMalloc(&d_A, h_A_sparse.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, h_B_col.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_E, 8 * sizeof(uint32_t)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A_sparse.data(), h_A_sparse.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_col.data(), h_B_col.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_E, h_E.data(), 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    sparse_gemm_kernel_tf32<<<1, 32>>>(d_A, d_B, d_C, d_E);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> h_C_gpu(m*n);
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, m*n*sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU Reference
    std::vector<float> h_C_ref(m*n, 0.0f);
    for(int r=0; r<m; ++r) for(int c=0; c<n; ++c) {
        float sum = 0;
        for(int l=0; l<k; ++l) if(l%4 < 2) sum += h_A_dense[r*k+l] * h_B[l*n+c];
        h_C_ref[r*n+c] = sum;
    }
    
    int errs = 0;
    for(int i=0; i<m*n; ++i) if(std::abs(h_C_gpu[i] - h_C_ref[i]) > 0.05f) errs++;
    if(errs == 0) std::cout << "Verification PASSED! (Definitive Mapping)" << std::endl;
    else {
        std::cout << "Verification FAILED: " << errs << " errors" << std::endl;
        for(int i=0; i<4; ++i) std::cout << h_C_gpu[i] << " vs " << h_C_ref[i] << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C)); CHECK_CUDA(cudaFree(d_E));
    return 0;
}
