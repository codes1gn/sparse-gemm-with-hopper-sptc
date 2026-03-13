# FP8 Sparse WGMMA Probe Archive

This folder collects the probe and reverse-engineering artifacts used to debug raw CUDA sparse FP8 WGMMA on Hopper.

The root of the repo is now kept clean. The old `tmp_*` files have been moved here because they are not product code, but they are still valuable engineering records.

There are two kinds of files here:

- `src/`: source files for one-off probes, experiments, and oracles
- `bin/`: reserved for rebuilt probe binaries if someone wants to regenerate them locally

The main narrative that ties these files together lives in `docs/raw-cuda-fp8-sparse-wgmma-debugging.md`.

## How to read this folder

The easiest way to understand the probes is to follow the same order a human debugger would follow.

### 1. Start with ground truth

- `src/oracle_cute_fp8_k64.cu`
  - A small CuTe oracle proving sparse FP8 `m64n8k64` works on real Hopper hardware.
  - This is the "is the operation even real?" probe.

### 2. Check static layout facts

- `src/probe_cute_fp8_layout_sizes.cpp`
  - Prints CuTe A/B/E layout storage sizes for FP8.
- `src/probe_sparse_metadata_sizes.cpp`
  - Compares sparse metadata atom and shared-memory sizes for F16 and FP8.
- `src/probe_gmma_descriptors.cpp`
  - Prints the descriptor fields implied by the official CuTe layouts.

These probes answer questions like:

- How large is the metadata allocation really?
- What leading and stride bytes does Hopper expect?
- Are we reasoning about the same tensor shape as CuTe?

### 3. Inspect logical offsets

- `src/probe_official_metadata_offsets.cpp`
  - Prints F16 and FP8 metadata raw offsets from the official layout.
- `src/probe_official_fp8_metadata_layout.cpp`
  - Focuses only on FP8 metadata raw layout.
- `src/probe_fp8_layout_reference_offsets.cpp`
  - Prints reference A/B layout offsets for FP8.
- `src/probe_fp8_layout_offsets.cu`
  - Another layout-centric offset dump for A/B/E views.

These are the probes that taught us the very important distinction between:

- logical raw coordinates
- physical shared-memory placement

### 4. Inspect metadata partition behavior

- `src/probe_e4m3_metadata_partition.cpp`
  - Early attempt to inspect per-thread metadata partitioning.
- `src/probe_cute_fp8_metadata_source_offsets.cu`
  - A more direct probe for how the CuTe metadata source view is partitioned for threads.
- `src/probe_manual_fp8_metadata_mapping.cu`
  - Probes the manually derived logical-to-physical metadata mapping.
- `src/probe_manual_fp8_metadata_registers.cu`
  - Probes the manual per-thread metadata register bytes.

These are the most reverse-engineering-heavy files in the folder. They exist because the core challenge was not just finding the right logical bytes, but finding how those bytes must be arranged so that each thread sees the right `u32` metadata fragment.

### 5. Use behavioral probes on the actual raw kernel

- `src/probe_raw_fp8_onehot_patterns.cu`
  - Onehot and alternating-pattern tests for the raw kernel.
- `src/probe_raw_fp8_single_tile_random.cu`
  - Small random-case probe for the raw kernel.
- `src/probe_raw_fp8_metadata_permutations.cu`
  - Brute-force experiments over byte permutations and bitwise transforms.

These files answer the practical question:

- Does the kernel still only work on easy patterns, or is it truly correct?

## Why these files matter

Most low-level tensor-core debugging has a short clean final patch and a long messy discovery path.

If we only kept the final patch, we would lose:

- the failed hypotheses
- the useful probes that can be reused later
- the evidence for why certain design decisions were made

That is why these probe files were archived instead of deleted.

## Current design status

At the end of this follow-up pass, the FP8 raw CUDA kernel path in `wgmma_sp_raw_common.hpp` no longer depends on CuTe for runtime metadata layout or per-thread metadata fragment construction.

The probe files remain useful because they document how that manual mapping was derived and validated.
