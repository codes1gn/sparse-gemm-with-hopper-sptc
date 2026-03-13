# Debugging Raw CUDA Sparse FP8 WGMMA on Hopper

This note is both a technique blog and a true worklog from the session that got raw CUDA sparse FP8 WGMMA working in this repo.

The goal was very specific:

- keep the real kernel path in raw CUDA with explicit `wgmma.mma_async.sp` PTX
- support Hopper sparse FP8 `e4m3`
- use the real Hopper sparse shapes, which for FP8 means `k64`, not a fake `k32`
- make it pass on real H800 hardware

The final result is good news first:

- raw sparse FP8 now passes on H800 for
  - `m64n8k64`
  - `m64n16k64`
  - `m64n32k64`
  - `m64n64k64`
  - `m64n128k64`
  - `m64n256k64`

But the path to the fix is the useful part, because the failure mode was subtle and very easy to misunderstand.

## The bug in one sentence

The raw FP8 kernel looked almost right, passed some friendly tests, and still failed random data because we mixed up three different things:

- FP8 sparse metadata encoding is not the same as the simple F16/BF16 nibble rule
- FP8 sparse metadata has a different logical layout than the older F16/BF16 mental model
- most importantly, FP8 sparse metadata must still live in the right swizzled shared-memory layout even if its logical coordinates look row-major

That last point was the trap.

## The human way to debug a problem like this

When a sparse WGMMA kernel compiles, runs, and produces wrong numbers, there are only a few big buckets to check:

1. Is the instruction itself real on this hardware?
2. Is the PTX spelling correct?
3. Is A in the right shared-memory layout?
4. Is B in the right shared-memory layout?
5. Is metadata E encoded correctly?
6. Is metadata E stored in shared memory correctly?
7. Is each thread reading the right metadata register fragment?

The session followed exactly this shape.

## Ground Truth First: what Hopper actually supports

Before touching code, we needed to stop guessing about shapes.

From the CUTLASS and CuTe headers, the real sparse Hopper support is:

- F16/BF16 sparse WGMMA: real `k32` shapes
- FP8 `e4m3` sparse WGMMA: real `k64` shapes, TN only

That means these FP8 sparse shapes are real:

- `m64n8k64`
- `m64n16k64`
- `m64n32k64`
- `m64n64k64`
- `m64n128k64`
- `m64n256k64`

And these old FP8 sparse `k32` demos were not the right target anymore:

- `mma_sp_wgmma_cuda_m64n8k32_fp32e4m3.cu`
- `mma_sp_wgmma_cuda_m64n32k32_fp32e4m3.cu`
- `mma_sp_wgmma_cuda_m64n64k32_fp32e4m3.cu`

This matters because a lot of wasted time in low-level tensor-core work comes from debugging an instruction shape that hardware never promised in the first place.

## Probe 1: Prove the instruction is real before debugging the raw kernel

### Idea

If the raw CUDA path is failing, the first question is: does the operation itself work at all on this GPU with this software stack?

### Method

Build a tiny CuTe oracle for one known-good sparse FP8 kernel.

The key line was this:

```cpp
using MmaOp = cute::SM90::GMMA::SPARSE::GMMA_64x8x64_F32E4M3E4M3_SS_TN<>;
```

That oracle lives in `tmp_cute_fp8_k64.cu`.

### Result

The CuTe kernel passed on H800.

### What we learned

This immediately eliminated three bad hypotheses:

- the hardware does support sparse FP8 `k64`
- the CUTLASS version in the repo is good enough
- the mathematical problem itself is not impossible

So the remaining bug had to be in our raw CUDA path.

## Probe 2: Compare the PTX spelling, not just the shape name

### Idea

It is common to blame inline PTX first. Sometimes that is right. This time it was mostly not.

### Method

We compared:

- the official PTX description for sparse F16 versus sparse FP8
- the CUTLASS CuTe inline assembly for FP8 sparse WGMMA
- our raw inline assembly in `wgmma_sp_raw_common.hpp`

The raw FP8 instruction shape was:

```cpp
"wgmma.mma_async.sp.sync.aligned.m64n8k64.f32.e4m3.e4m3 "
```

And the important semantic checks were:

- metadata operand is a 32-bit register
- `sp-sel` is an immediate
- for FP8 `m64nNk64`, `sp-sel = 0`
- `scale_d` controls zero-versus-accumulate
- `scale-a` and `scale-b` are the usual sign/scale immediates

### Result

The raw PTX string and control operands were already structurally correct.

### What we learned

This was a strong hint that the bug was not the instruction spelling. It pushed the investigation toward layout and metadata semantics.

## Probe 3: Check A first, because a wrong A layout can fake a metadata bug

### Idea

When A is sparse, it is easy to blame metadata. But if the compressed A payload is stored in the wrong shared-memory layout, metadata can look guilty even when it is innocent.

### Method

We built offset probes and compared our raw mapping against CuTe's sparse K-interleaved layout.

The old mapping was effectively too simple for FP8 `k64`.

Wrong mental model:

```cpp
// old idea
row * 32 + col
```

The corrected mapping became:

```cpp
__device__ inline int smem_a_index_k64_e4m3(int row, int col) {
  return (col & 15) + row * 16 + ((col >> 4) * 1024);
}
```

This is now in `wgmma_sp_raw_common.hpp`.

### Result

The A layout fix improved behavior and matched the CuTe reference layout.

### What we learned

The raw A path had been wrong for FP8 `k64`. Fixing it was necessary, but not sufficient.

## Probe 4: Check B next, but do not overfit a bug that is not there

### Idea

B is dense, but FP8 still uses Hopper's interleaved K-major shared-memory layout. If B is wrong, everything is wrong.

### Method

We compared our raw B indexing with CuTe's `Layout_K_INTER_Atom` behavior.

The raw index used was:

```cpp
template <int BlockN>
__device__ inline int smem_b_index_k64_e4m3(int col, int kk) {
  return (kk & 15) + col * 16 + ((kk >> 4) * (BlockN * 16));
}
```

### Result

B already matched the canonical layout.

### What we learned

Do not keep changing a part just because the whole kernel is still failing. B was not the interesting bug.

## Probe 5: Study metadata E logically before touching thread registers

### Idea

Metadata bugs are hard because there are two layers:

- the logical coordinates of metadata bytes
- the physical shared-memory placement that makes warpgroup loads line up

Humans naturally mix those up.

### Method

We wrote official-layout probes using CuTe and CUTLASS to print raw metadata byte offsets.

The probing pattern in `tmp_official_e_offsets.cpp` was basically:

```cpp
auto fp8sE = make_tensor(make_smem_ptr(recast_ptr<Fp8E>(nullptr)), Fp8SmemE{});
auto fp8raw = recast<uint8_t>(fp8sE);

for (int r = 0; r < 16; ++r) {
  for (int c = 0; c < 8; ++c) {
    std::cout << fp8raw.layout()(make_coord(r, c));
  }
}
```

### Result

For FP8 sparse `k64`, the logical raw metadata tensor looked like a simple row-major `64 x 8` byte matrix:

- row 0: `0 1 2 3 4 5 6 7`
- row 1: `8 9 10 11 12 13 14 15`
- and so on

### What we learned

This killed the old hand-written FP8 metadata permutation formula. We changed the logical FP8 raw indexing to:

```cpp
__device__ inline int smem_e_index_k64_e4m3(int row, int byte_col) {
  return row * 8 + byte_col;
}
```

But that was still only half the story.

## Probe 6: Use easy patterns and hard patterns on purpose

### Idea

Friendly tests are useful, but they lie. A kernel can pass a structured pattern and still be deeply wrong.

### Method

We built targeted probes:

- `tmp_raw_fp8_onehot.cu`
- `tmp_run_fp8_single_tile.cu`
- `tmp_raw_fp8_meta_perm.cu`

The onehot cases were intentionally simple: make B activate a narrow K slice so the expected output reveals where each sparse pair is landing.

The alternating pattern was intentionally mean: cycle through all six valid 2:4 pairs so bad metadata handling becomes visible quickly.

### Result before the final fix

- onehot/simple cases could pass
- alternating-pattern cases failed badly
- random cases failed badly, often with nearly every output wrong

Typical symptoms were:

- alternating onehot cases failing with `128`, `160`, or `192` errors
- random `64x8x64` failing with `511` errors

### What we learned

This was a huge clue. It meant:

- the top-level kernel was not completely broken
- descriptors were probably okay
- some easy metadata cases accidentally lined up
- mixed metadata cases did not

In other words: this still smelled like metadata semantics.

## Probe 7: The metadata nibble itself is different for FP8

### Idea

Our generic host compressor used the obvious nibble formula:

```cpp
idx0 | (idx1 << 2)
```

That is tempting, compact, and wrong for Hopper sparse 8-bit metadata.

### Method

We inspected CUTLASS legacy sparse compressor logic in `third_party/cutlass/test/unit/transform/device/sm90_sparse_gemm_compressor_legacy.hpp`.

The legal Hopper metadata encodings for 8-bit sparse pairs are:

- `(0,1) -> 0x4`
- `(1,2) -> 0x9`
- `(2,3) -> 0xE`
- `(0,2) -> 0x8`
- `(1,3) -> 0xD`
- `(0,3) -> 0xC`

We updated the host compressor in `wgmma_sp_raw_common.hpp` accordingly.

The new code path looks like this:

```cpp
if constexpr (std::is_same_v<Element, __nv_fp8_e4m3>) {
  if (idxs[0] == 0 && idxs[1] == 1) nibble = 0x4;
  else if (idxs[0] == 1 && idxs[1] == 2) nibble = 0x9;
  else if (idxs[0] == 2 && idxs[1] == 3) nibble = 0xE;
  else if (idxs[0] == 0 && idxs[1] == 2) nibble = 0x8;
  else if (idxs[0] == 1 && idxs[1] == 3) nibble = 0xD;
  else if (idxs[0] == 0 && idxs[1] == 3) nibble = 0xC;
}
```

### Result

Important but incomplete. It was necessary, but the raw kernel still was not fully correct.

### What we learned

This is exactly the kind of bug that tricks you: a real fix can still leave the system failing because another bug is stacked on top of it.

## The key insight: logical row-major is not the same as physical shared-memory placement

This is the main lesson of the session.

At one point we had two facts in hand:

1. the logical FP8 metadata tensor `sEraw(row, byte_col)` is row-major
2. the raw kernel still failed even after fixing the nibble encoding

That seemed contradictory until we separated logical coordinates from physical placement.

### The wrong mental model

We treated FP8 metadata as if this were enough:

```cpp
shared.smem_E[row * 8 + byte_col] = e_bytes[...];
```

That writes a flat byte array.

### The subtle reality

CuTe's working path does not just say where the logical metadata byte lives. It also builds the swizzled sparse shared-memory tensor and then forms each thread's metadata register fragment from that tensor.

So for FP8 sparse WGMMA, these are different questions:

- What is the logical coordinate of metadata byte `(row, byte_col)`?
- What physical shared-memory address must hold that byte so the warpgroup's `u32` metadata fragment loads come out right?

The logical answer was row-major.

The physical answer still required the official sparse shared-memory layout machinery.

## The final fix

The fix that finally made FP8 pass combined three things.

### 1. Use the legal FP8 metadata nibble encoding

Already described above. Without this, mixed 2:4 patterns are wrong.

### 2. Allocate FP8 metadata shared memory using the real sparse layout size

For FP8, the shared-memory metadata storage is not just `64 * 8 = 512` bytes in the way we were thinking about it.

The code now uses the official sparse layout cosize:

```cpp
using RawFp8MetadataSmemLayoutE = decltype(cute::tile_to_shape(
    RawFp8MetadataSmemLayoutAtomE{},
    cute::Shape<cute::_64, cute::_64>{}));

constexpr int kRawFp8MetadataSmemBytes = cute::cosize_v<RawFp8MetadataSmemLayoutE>;
```

And the FP8 shared-storage size is wired through:

```cpp
static constexpr int kESmemBytes = kRawFp8MetadataSmemBytes;
```

### 3. Store metadata through the official logical tensor view, then derive the per-thread `u32` fragment from that layout

This was the decisive part.

Instead of writing FP8 metadata as a plain byte array, the kernel now does:

```cpp
auto sE = raw_fp8_smem_e_tensor(shared.smem_E);
auto sEraw = cute::recast<uint8_t>(sE);

for (int idx = tid; idx < kRawBlockM * kMetaBytes; idx += kRawThreads) {
  int row = idx / kMetaBytes;
  int byte_col = idx % kMetaBytes;
  sEraw(row, byte_col) = e_bytes[(block_row + row) * (k / 8) + (k_tile / 8) + byte_col];
}

uint32_t e = raw_metadata_u32<BlockN, Element>(e_bytes, shared.smem_E, block_row, k_tile, k, tid);
```

And that metadata register is built by the per-shape helper:

```cpp
template <int BlockN>
__device__ inline uint32_t raw_fp8_metadata_u32(uint8_t* smem_e, int tid) {
  using MmaOp = typename RawFp8MetadataMmaOp<BlockN>::type;
  using TiledMma = decltype(cute::make_tiled_mma(MmaOp{}));
  ...
  auto rE = cute::recast<uint32_t>(tCrE);
  return rE[0];
}
```

This exactly matched the working CuTe behavior closely enough to fix the kernel.

## Why the final fix makes sense in hindsight

Once the kernel passed, the earlier confusing observations suddenly fit together:

- onehot cases passed because simple metadata patterns can survive a surprising amount of wrong packing
- alternating cases failed because they stress the true metadata semantics
- random cases failed because they amplify both encoding and placement mistakes
- the old row-major byte store was only logically right, not physically right for the per-thread metadata fragment formation

This is very common in tensor-core debugging: the bug is not in the math, but in the difference between a logical tensor view and the exact hardware consumption pattern.

## What passed on real H800 hardware

After the final fix, these probes and demos passed.

### Focused probes

- `tmp_raw_fp8_onehot`
  - all onehot cases passed
  - all alternating-pattern onehot cases passed
- `tmp_run_fp8_single_tile`
  - `random_64x8x64` passed
  - `random_64x8x128` passed

### Raw CUDA demo binaries

- `mma_sp_wgmma_cuda_m64n8k64_fp32e4m3`
- `mma_sp_wgmma_cuda_m64n16k64_fp32e4m3`
- `mma_sp_wgmma_cuda_m64n32k64_fp32e4m3`
- `mma_sp_wgmma_cuda_m64n64k64_fp32e4m3`
- `mma_sp_wgmma_cuda_m64n128k64_fp32e4m3`
- `mma_sp_wgmma_cuda_m64n256k64_fp32e4m3`

All of them passed their pattern and random verification cases on H800.

## True worklog from this session

This is the real sequence of ideas and results, not a cleaned-up fairy tale.

1. Confirmed the target shapes from CUTLASS and PTX: FP8 sparse means real Hopper `k64` TN shapes.
2. Verified the raw larger-N F16/BF16 CUDA kernels already worked on H800.
3. Compared official PTX sparse F16 and sparse FP8 descriptions and concluded the raw FP8 opcode spelling was probably fine.
4. Built a CuTe FP8 sparse `k64` oracle and proved the operation worked on hardware.
5. Probed A layout and fixed the FP8 sparse A shared-memory mapping.
6. Probed E layout and learned that logical FP8 metadata bytes are row-major `64 x 8`.
7. Changed the old hand-written FP8 logical E index accordingly.
8. Re-ran probes and found an important pattern: simple onehot passed, alternating and random still failed.
9. Inspected the legacy CUTLASS compressor and discovered the legal 8-bit metadata nibble codes.
10. Implemented the FP8-specific nibble encoding in the host compressor.
11. Re-ran probes and still saw failures, which meant encoding alone was not the whole story.
12. Returned to the CuTe oracle and realized the real missing piece: FP8 metadata must still be stored through the official sparse shared-memory tensor, not a flat byte buffer.
13. Changed FP8 `smem_E` sizing and writes to use the CuTe sparse metadata layout and per-thread fragment construction.
14. Re-ran the onehot, alternating, single-tile random, and all final FP8 demo kernels on H800.
15. Everything passed.

That is the real shape of low-level debugging: not one clever jump, but a sequence of small eliminations until the remaining explanation is finally precise enough.

## Reusable probe techniques from this work

These techniques are worth keeping for future low-level tensor-core work.

### 1. Build a small working oracle first

If you can make the same operation pass through CuTe or CUTLASS, you have a stable reference for:

- instruction legality
- layout expectations
- metadata fragment behavior

### 2. Probe offsets, not just values

For shared-memory layout bugs, printing offsets often teaches more than printing tensor values.

### 3. Use intentionally simple B patterns

Onehot or narrow-band B patterns make sparse metadata mistakes visible almost immediately.

### 4. Separate logical layout from physical storage

This session only finished once we stopped assuming those two were the same thing.

### 5. Keep one adversarial structured test

The alternating 2:4 pair pattern was much more informative than a friendly pattern case.

## Design note: is the current FP8 `_cuda_` path fully manual and pure CUDA?

Not yet.

The current FP8 raw CUDA path is functionally correct, and the actual WGMMA instruction is still emitted through explicit inline PTX. But from a design-purity point of view, the FP8 metadata path still leans on CuTe.

Today the FP8 path still uses CuTe for:

- the sparse metadata shared-memory layout type
- the per-thread metadata partitioning logic
- the copy that forms each thread's `u32` metadata fragment

By contrast, the F16/BF16 raw CUDA path is closer to a fully manual design:

- manual shared-memory index formulas
- manual metadata byte addressing
- manual `ld.shared.u32`
- raw PTX WGMMA call

So the honest answer is:

- the FP8 kernel is correct and raw at the instruction level
- but it is not yet as purely manual as the F16 path

## The next engineering step

If the design goal is that the FP8 `_cuda_` kernels should have exactly the same style as F16, then the next step is clear:

- replace the current CuTe-backed FP8 metadata helper path with explicit pure-CUDA logic
- derive the exact FP8 metadata shared-memory swizzle and per-thread `u32` fragment mapping manually
- keep the current verified behavior as the reference
- preserve the same raw PTX instruction specializations that are already passing

In other words, the next refactor is not about correctness anymore. It is about design cleanup:

- same answers
- same hardware behavior
- less CuTe in the FP8 raw path
- a more uniform "pure CUDA" story across F16, BF16, and FP8

That is the right next step now that the kernel is finally correct.
