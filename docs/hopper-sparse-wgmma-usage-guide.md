# Hopper Sparse WGMMA Raw CUDA Usage Guide

This document is a complete reference for using the raw CUDA sparse WGMMA kernels in this repository. It covers everything from high-level concepts to low-level kernel integration.

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Supported Shapes and Data Types](#supported-shapes-and-data-types)
4. [Architecture Overview](#architecture-overview)
5. [Quick Start](#quick-start)
6. [The Host-Side API](#the-host-side-api)
7. [The Kernel API](#the-kernel-api)
8. [Shared Memory Layouts](#shared-memory-layouts)
9. [Metadata Encoding](#metadata-encoding)
10. [Creating a New Kernel Demo](#creating-a-new-kernel-demo)
11. [Build System](#build-system)
12. [Verification and Testing](#verification-and-testing)
13. [Performance Considerations](#performance-considerations)
14. [Common Pitfalls](#common-pitfalls)
15. [Integration Example](#integration-example)

---

## Overview

This repository provides raw CUDA implementations of Hopper's sparse WGMMA (`wgmma.mma_async.sp`) instructions. The implementations:

- Use explicit inline PTX for the `wgmma.mma_async.sp` instruction
- Manually manage shared memory layouts for A, B, and E (metadata)
- Support fp16, bf16, and fp8 e4m3 data types
- Use the real Hopper sparse shapes (not emulated)
- Are verified on real NVIDIA H800 hardware

The code is organized into two tiers:

1. **Production header**: `wgmma_sp_raw_common.hpp` — the actual implementation
2. **Reference header**: `wgmma_sp_demo_common.hpp` — CuTe-based reference for validation

---

## Hardware Requirements

- **GPU**: NVIDIA Hopper (sm_90a)
  - Tested on: H800 PCIe
- **CUDA Toolkit**: 12.0 or later recommended
- **Driver**: R525 or later

The kernel will abort gracefully on pre-Hopper hardware:

```cpp
if (props.major < 9) {
  std::cout << "This demo requires Hopper-class hardware." << std::endl;
  return 0;
}
```

---

## Supported Shapes and Data Types

### FP16 and BF16 (k32)

| Shape | BlockK | Sparse K | Notes |
|-------|--------|----------|-------|
| m64n8k32 | 32 | 16 | Minimum N |
| m64n16k32 | 32 | 16 | |
| m64n32k32 | 32 | 16 | |
| m64n64k32 | 32 | 16 | |
| m64n128k32 | 32 | 16 | |
| m64n256k32 | 32 | 16 | Maximum N verified |

### FP8 E4M3 (k64)

| Shape | BlockK | Sparse K | Notes |
|-------|--------|----------|-------|
| m64n8k64 | 64 | 32 | Minimum N |
| m64n16k64 | 64 | 32 | |
| m64n32k64 | 64 | 32 | |
| m64n64k64 | 64 | 32 | |
| m64n128k64 | 64 | 32 | |
| m64n256k64 | 64 | 32 | Maximum N verified |

**Important**: There is no valid sparse FP8 `k32` shape on Hopper. The FP8 sparse WGMMA instruction only supports `k64`.

---

## Architecture Overview

### The 2:4 Sparse Format

Hopper sparse WGMMA uses the 2:4 sparse format:

- Every group of 4 consecutive elements in the K dimension contains exactly 2 non-zero values
- The metadata (E tensor) encodes which 2 of the 4 elements are non-zero
- The A matrix is stored in compressed form (2 elements per group of 4)
- The B matrix is dense

### Warpgroup Structure

Each sparse WGMMA tile operates on:

- **M dimension**: 64 rows (fixed by hardware)
- **N dimension**: 8, 16, 32, 64, 128, or 256 (variable)
- **K dimension**: 32 (fp16/bf16) or 64 (fp8)

The kernel uses one warpgroup (128 threads) per tile. The tile is processed by having threads cooperatively load A, B, and E into shared memory, then issuing the WGMMA instruction.

---

## Quick Start

### Build and Run an Existing Demo

```bash
# Build all demos (excludes fp8 by default)
make

# Build including fp8 demos
make ENABLE_FP8=1

# Run a specific demo
make KERNEL=mma_sp_wgmma_cuda_m64n64k32_fp32fp16 run

# Run fp8 demo
make KERNEL=mma_sp_wgmma_cuda_m64n8k64_fp32e4m3 run

# List available demos
make help
```

### Verify on H800

All demos pass verification against a CPU reference implementation:

```
Running mma_sp_wgmma_cuda_m64n64k32_fp32fp16 on CUDA device 1...
pattern_64x64x32 total errors: 0
random_64x64x128 total errors: 0
random_128x128x256 total errors: 0
Verification PASSED
```

---

## The Host-Side API

### RawSparseConfig

Each element type has a specialization that defines the shape constants:

```cpp
template <>
struct RawSparseConfig<half> {
  static constexpr int kBlockK = 32;        // K tiles are 32 for fp16
  static constexpr int kSparseK = 16;       // Compressed A is K/2
  static constexpr int kMetaBytes = 4;      // Metadata is K/8 bytes per row
  static constexpr int kASmemElements = 64 * 16;  // 64 x 16 compressed elements
  static constexpr int kESmemBytes = 64 * 4;     // 64 x 4 metadata bytes
  static constexpr uint32_t kADescLeading = 64;
  static constexpr uint32_t kADescStride = 8;
};

template <>
struct RawSparseConfig<__nv_fp8_e4m3> {
  static constexpr int kBlockK = 64;        // K tiles are 64 for fp8
  static constexpr int kSparseK = 32;
  static constexpr int kMetaBytes = 8;
  static constexpr int kASmemElements = 64 * 32;
  static constexpr int kESmemBytes = kRawFp8MetadataSmemBytes;  // Swizzled layout
  static constexpr uint32_t kADescLeading = 64;
  static constexpr uint32_t kADescStride = 8;
};
```

### Compression

Before calling the kernel, the A matrix must be compressed from dense to 2:4 sparse format:

```cpp
std::vector<half> a_dense(m * k);
std::vector<half> a_sparse(m * (k / 2));
std::vector<uint8_t> e_bytes(m * (k / 8));

// Initialize with 2:4 structured sparsity
init_structured_sparse_a(a_dense, m, k, gen);

// Compress
compress_structured_sparse_a(a_dense, a_sparse, e_bytes, m, k);
```

The compression function:
1. Identifies the two largest-magnitude elements in each group of 4
2. Stores those two elements in the compressed A buffer
3. Encodes the indices in the metadata E buffer using the Hopper-specific nibble encoding

### Kernel Invocation

```cpp
dim3 block(kRawThreads);        // 128 threads
dim3 grid(n / BlockN, m / 64);  // One block per output tile
wgmma_sp_raw_kernel<Element, BlockN><<<grid, block>>>(
    d_a, d_b, d_e, d_c, n, k);
```

---

## The Kernel API

### Kernel Template

```cpp
template <int BlockN, typename Element>
__global__ void wgmma_sp_raw_kernel(
    Element const* __restrict__ a_sparse,  // Compressed A (M x K/2)
    Element const* __restrict__ b,         // Dense B (K x N)
    uint8_t const* __restrict__ e_bytes,   // Metadata (M x K/8)
    float* __restrict__ c,                 // Output (M x N)
    int n,                                  // N dimension
    int k);                                 // K dimension
```

### Thread Requirements

- **M dimension**: Must be a multiple of 64
- **N dimension**: Must be a multiple of BlockN (8, 16, 32, 64, 128, or 256)
- **K dimension**: Must be a multiple of 32 (fp16/bf16) or 64 (fp8)

### Kernel Launch

```cpp
wgmma_sp_raw_kernel<half, 64><<<grid, block>>>(
    d_a_sparse, d_b, d_e, d_c, n, k);
```

---

## Shared Memory Layouts

The shared memory holds three buffers: A, B, and E. Each has a specific layout that matches Hopper's GMMA descriptor expectations.

### A Matrix (Sparse, Compressed)

The A matrix is stored in K-interleaved format:

```cpp
// Fp16/BF16: 64 rows x 16 compressed columns
__device__ inline int smem_a_index(int row, int col) {
  return (col & 7) + ((row >> 3) * 64) + ((row & 7) * 8) + ((col >> 3) * 512);
}

// FP8 E4M3: 64 rows x 32 compressed columns (k64)
__device__ inline int smem_a_index_k64_e4m3(int row, int col) {
  return (col & 15) + row * 16 + ((col >> 4) * 1024);
}
```

### B Matrix (Dense)

The B matrix is stored in K-interleaved format:

```cpp
template <int BlockN>
__device__ inline int smem_b_index(int col, int kk) {
  return (col & 7) + ((col >> 3) * 64) + ((kk >> 3) * (BlockN * 8)) + ((kk & 7) * 8);
}

template <int BlockN>
__device__ inline int smem_b_index_k64_e4m3(int col, int kk) {
  return (kk & 15) + col * 16 + ((kk >> 4) * (BlockN * 16));
}
```

### E Matrix (Metadata)

The metadata encoding is the trickiest part. The logical layout is row-major, but Hopper requires a specific physical swizzle for the per-thread `u32` fragment loads.

#### Fp16/BF16 Metadata

```cpp
// k32: 64 rows x 4 bytes
__device__ inline int smem_e_index(int row, int byte_col) {
  return (byte_col & 1) + ((byte_col >> 1) * 32) + ((row >> 4) * 64) + ((row & 7) * 4) + (((row >> 3) & 1) * 2);
}

// k64: 64 rows x 8 bytes
__device__ inline int smem_e_index_k64(int row, int byte_col) {
  return (byte_col & 1) + ((byte_col >> 1) * 32) + ((row >> 4) * 128) + ((row & 7) * 4) + (((row >> 3) & 1) * 2);
}
```

#### FP8 E4M3 Metadata (Manual Swizzle)

```cpp
// This is the manual swizzle that replaces the CuTe path
__device__ inline int smem_e_index_k64_e4m3(int row, int byte_col) {
  int row_block = row >> 4;
  int row_in_block = row & 15;
  int row_lo = row_in_block & 7;
  int row_hi = row_in_block >> 3;
  int col_group = byte_col >> 2;
  int col_lo = byte_col & 3;
  return row_block * 128 + col_group * 64 + row_lo * 8 + row_hi * 4 + col_lo;
}
```

---

## Metadata Encoding

### The 2:4 Nibble Encoding

Hopper sparse metadata uses a 4-bit nibble per group of 4 elements to encode which 2 are non-zero.

#### Fp16/BF16 Encoding

Simple bit packing works:

```cpp
nibble = idx0 | (idx1 << 2);
```

Where `idx0` and `idx1` are the indices (0-3) of the two non-zero elements.

#### FP8 E4M3 Encoding

FP8 uses different legal codes (from CUTLASS legacy compressor):

```cpp
if (idx0 == 0 && idx1 == 1) nibble = 0x4;
else if (idx0 == 1 && idx1 == 2) nibble = 0x9;
else if (idx0 == 2 && idx1 == 3) nibble = 0xE;
else if (idx0 == 0 && idx1 == 2) nibble = 0x8;
else if (idx0 == 1 && idx1 == 3) nibble = 0xD;
else if (idx0 == 0 && idx1 == 3) nibble = 0xC;
```

These are not arbitrary — they are the specific codes that Hopper's sparse WGMMA instruction accepts.

### Per-Thread Metadata Fragment

Each thread in the warpgroup reads a 32-bit metadata fragment. The fragment is formed by loading 4 consecutive metadata bytes from shared memory:

```cpp
// Per-thread byte offset for k32
__device__ inline int e_thread_byte_offset(int tid) {
  return ((tid & 1) * 32) + ((tid >> 5) * 64) + (((tid >> 2) & 7) * 4);
}

// Per-thread byte offset for fp8 k64 (manual derivation)
__device__ inline int e_thread_byte_offset_k64_e4m3(int tid) {
  int row = ((tid >> 2) & 7) + ((tid & 1) << 3) + ((tid >> 5) << 4);
  int byte_col = ((tid >> 1) & 1) << 2;
  return smem_e_index_k64_e4m3(row, byte_col);
}
```

The actual load is:

```cpp
uint32_t e = ld_shared_u32(smem_e + raw_e_thread_byte_offset<Element>(tid));
```

---

## Creating a New Kernel Demo

### Step 1: Create the .cu File

Create a new file like `mma_sp_wgmma_cuda_m64n128k32_fp32bf16.cu`:

```cpp
#include "wgmma_sp_raw_common.hpp"

using Demo = RawWgmmaSparseDemo<__nv_bfloat16, 128>;

int main() {
  constexpr RawDemoCase kCases[] = {
      {"pattern_64x128x32", 64, 128, 32, true},
      {"random_64x128x256", 64, 128, 256, false},
      {"random_128x256x256", 128, 256, 256, false},
  };
  return run_raw_demo_suite<Demo>(
      "wgmma.mma_async.sp.sync.aligned.m64n128k32.f32.bf16.bf16",
      kCases);
}
```

### Step 2: Build and Run

```bash
make mma_sp_wgmma_cuda_m64n128k32_fp32bf16
./mma_sp_wgmma_cuda_m64n128k32_fp32bf16
```

### Template Parameters

| Parameter | Description | Valid Values |
|-----------|-------------|--------------|
| `Element` | Data type | `half`, `__nv_bfloat16`, `__nv_fp8_e4m3` |
| `BlockN` | N tile size | 8, 16, 32, 64, 128, 256 |

---

## Build System

### Makefile Targets

```bash
# Build all non-fp8 demos
make

# Build all demos including fp8
make ENABLE_FP8=1

# Build specific demo
make mma_sp_wgmma_cuda_m64n64k32_fp32fp16

# Run specific demo
make KERNEL=mma_sp_wgmma_cuda_m64n64k32_fp32fp16 run

# Run on specific GPU
make CUDA_ID=0 KERNEL=mma_sp_wgmma_cuda_m64n64k32_fp32fp16 run

# Clean
make clean

# List available demos
make help
```

### Compilation Flags

- **Architecture**: `sm_90a` (required for WGMMA)
- **Standard**: C++17 for WGMMA kernels
- **Optimization**: `-O3`

---

## Verification and Testing

### Built-in Verification

Every demo kernel includes automatic verification:

1. **Pattern tests**: Structured input that produces known outputs
2. **Random tests**: Random input compared against CPU reference

```cpp
// Pattern test example
init_pattern_sparse_a(a_dense, m, k);
init_pattern_b(b, k, n);

// Random test example  
init_structured_sparse_a(a_dense, m, k, gen);
init_random_b(b, k, n, gen);
```

### Verification Tolerance

The default absolute tolerance is `1e-2`:

```cpp
verify_result(tag, c_gpu, c_ref, n, 1.0e-2f);
```

### Error Reporting

On failure, the first 8 mismatches are printed:

```
pattern_64x64x32 mismatch at (0, 32) gpu=1.234 ref=1.250 diff=0.016
```

---

## Performance Considerations

### Occupancy

- Each kernel uses 128 threads (one warpgroup)
- Shared memory size varies by shape:
  - FP16 k32 n64: ~12KB
  - FP8 k64 n64: ~20KB
- Maximum occupancy is limited by shared memory

### Memory Access Patterns

- A and B use K-interleaved layouts for optimal GMMA descriptor performance
- E uses a swizzled layout that aligns with per-thread `u32` fragment loads
- All loads are coalesced within the warpgroup

### Instruction Latency

- `wgmma.mma_async.sp` is a asynchronous instruction
- It is issued with `fma` and requires `wgmma.commit_group.sync.aligned`
- The following `wgmma.wait_group.sync.aligned` ensures completion

```cpp
warpgroup_fence_accum(accum);
warpgroup_arrive();
RawSparseWgmma<BlockN, Element>::fma(desc_a, desc_b, accum, e, scale_d);
warpgroup_commit_batch();
warpgroup_wait<0>();
warpgroup_fence_accum(accum);
```

---

## Common Pitfalls

### 1. Wrong K Dimension for FP8

**Wrong**: Trying to use `k32` with FP8
**Right**: FP8 requires `k64`

```cpp
// This will fail
using Demo = RawWgmmaSparseDemo<__nv_fp8_e4m3, 64>;  // k=32 is invalid for fp8

// This is correct
constexpr int k = 64;  // Must be multiple of 64
```

### 2. M Dimension Not Multiple of 64

**Wrong**: `m = 32`
**Right**: `m % 64 == 0`

### 3. N Dimension Not Multiple of BlockN

**Wrong**: `n = 100` with `BlockN = 64`
**Right**: `n % BlockN == 0`

### 4. Wrong Metadata Nibble Encoding

**Wrong**: Using simple bit packing for FP8
**Right**: Use the Hopper-specific FP8 nibble codes

### 5. Forgetting to Compress A

**Wrong**: Passing dense A to the kernel
**Right**: Compress with `compress_structured_sparse_a()`

### 6. Not Using the Correct Shared Memory Swizzle

**Wrong**: Using simple row-major for E
**Right**: Use the explicit swizzle formulas for E

---

## Integration Example

Here is a minimal example of integrating the sparse WGMMA kernel into your own code:

```cpp
#include "wgmma_sp_raw_common.hpp"

template <typename Element, int BlockN>
void my_sparse_gemm(
    Element const* A,    // Compressed sparse A (M x K/2)
    Element const* B,    // Dense B (K x N)
    uint8_t const* E,   // Metadata (M x K/8)
    float* C,            // Output (M x N)
    int M, int N, int K) {
  
  Element* d_A;
  Element* d_B;
  uint8_t* d_E;
  float* d_C;
  
  size_t a_size = M * (K / 2) * sizeof(Element);
  size_t b_size = K * N * sizeof(Element);
  size_t e_size = M * (K / 8);
  size_t c_size = M * N * sizeof(float);
  
  CHECK_CUDA(cudaMalloc(&d_A, a_size));
  CHECK_CUDA(cudaMalloc(&d_B, b_size));
  CHECK_CUDA(cudaMalloc(&d_E, e_size));
  CHECK_CUDA(cudaMalloc(&d_C, c_size));
  
  CHECK_CUDA(cudaMemcpy(d_A, A, a_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, b_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_E, E, e_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_C, 0, c_size));
  
  dim3 block(128);
  dim3 grid(N / BlockN, M / 64);
  wgmma_sp_raw_kernel<Element, BlockN><<<grid, block>>>(
      d_A, d_B, d_E, d_C, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  
  CHECK_CUDA(cudaMemcpy(C, d_C, c_size, cudaMemcpyDeviceToHost));
  
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_E));
  CHECK_CUDA(cudaFree(d_C));
}
```

Then call it:

```cpp
// Example: fp16, m64n64k32
std::vector<half> A_sparse(M * K / 2);
std::vector<half> B(K * N);
std::vector<uint8_t> E(M * K / 8);
std::vector<float> C(M * N);

// ... populate A_sparse, B, E with 2:4 structured sparsity ...

my_sparse_gemm<half, 64>(A_sparse.data(), B.data(), E.data(), C.data(), M, N, K);
```

---

## Summary

This implementation provides:

1. **Real Hopper sparse WGMMA**: Uses actual `wgmma.mma_async.sp` PTX, not emulated
2. **Full shape coverage**: All valid fp16/bf16 (k32) and fp8 e4m3 (k64) shapes
3. **Manual shared memory**: Explicit index formulas for A, B, and E
4. **Correct metadata**: Both the encoding and the physical placement
5. **Verified**: All demos pass on real H800 hardware

The key insight is that sparse tensor core programming requires precise attention to:
- The 2:4 sparse format and its metadata encoding
- The K-interleaved shared memory layouts
- The per-thread metadata fragment formation
- The asynchronous WGMMA instruction pipeline

With these primitives, you can build high-performance sparse matrix multiplication on Hopper.
