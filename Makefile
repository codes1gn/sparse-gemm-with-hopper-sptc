
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch=sm_80
ARCH_FP8 = -arch=sm_90a
FLAGS = -O3 -std=c++14
CUTLASS_INCLUDES = -Ithird_party/cutlass/include -Ithird_party/cutlass/tools/util/include
FLAGS_WGMMA = -O3 -std=c++17 $(CUTLASS_INCLUDES)

AUTO_CUDA_ID := $(shell nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -t, -k2 -nr | head -n1 | cut -d, -f1)

# Default CUDA device id (auto-selects GPU with most free memory; can be overridden)
# Example: `make CUDA_ID=0 run-foo` or `make CUDA_ID=0 RUN_BIN=foo run`
CUDA_ID ?= $(if $(AUTO_CUDA_ID),$(AUTO_CUDA_ID),0)

# Default binary to run with `make run` (can be overridden)
KERNEL ?= mma_sp_m16n8k32_fp32fp16

# Discover only mma_sp_* CUDA sources so each kernel gets its own binary
# FP8 sparse MMA is not supported by the PTX ISA today; exclude by default.
BASE_SRCS := $(wildcard mma_sp_*.cu)
FP8_SRCS := $(wildcard mma_sp_*fp8.cu)
SRCS := $(filter-out $(FP8_SRCS),$(BASE_SRCS))

# Enable FP8 build explicitly (may still fail if ISA lacks sparse FP8 support)
ifeq ($(ENABLE_FP8),1)
	SRCS := $(BASE_SRCS)
endif
BINS := $(SRCS:.cu=)

# Map numeric KERNEL to binary name (1=first, 2=second, etc.)
# Logic: If KERNEL is a number, use word, else use as name.
IS_NUM := $(shell echo $(KERNEL) | grep -E '^[0-9]+$$')
BIN_TO_RUN := $(strip $(if $(IS_NUM), $(word $(KERNEL), $(BINS)), $(KERNEL)))

.PHONY: all clean run run-all info list
all: $(BINS)

# Pattern rule: compile %.cu -> %
%: %.cu
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $<

# FP8 and WGMMA kernels require Hopper; build them with sm_90a.
mma_sp_%fp8: ARCH := $(ARCH_FP8)
mma_sp_wgmma_%: ARCH := $(ARCH_FP8)
mma_sp_wgmma_%: FLAGS := $(FLAGS_WGMMA)

# Default run: runs the configured BIN_TO_RUN (must exist)
run: $(BIN_TO_RUN)
	@echo "Running $(BIN_TO_RUN) on CUDA device $(CUDA_ID)..." \
	&& CUDA_VISIBLE_DEVICES=$(CUDA_ID) ./$(BIN_TO_RUN)

# List available run targets with indices
help:
	@echo "Available binaries:"
	@n=1; for b in $(BINS); do echo "  $$n. $$b"; n=$$((n+1)); done
	@echo ""
	@echo "Run a binary: make KERNEL=<name_or_id> run"
	@echo "Example: make KERNEL=1 run"

# Print discovery / config info
info: ; @echo "SRCS = $(SRCS)"
	@echo "BINS = $(BINS)"
	@echo "KERNEL = $(KERNEL) (BIN_TO_RUN = $(BIN_TO_RUN))"
	@echo "CUDA_ID = $(CUDA_ID) (override with 'make CUDA_ID=0')"

clean:
	rm -f $(BINS)
