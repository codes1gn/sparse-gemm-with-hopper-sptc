
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch=sm_80
ARCH_FP8 = -arch=sm_90a
FLAGS = -O3 -std=c++14

# Default CUDA device id (can be overridden on the make command line)
# Example: `make CUDA_ID=0 run-foo` or `make CUDA_ID=0 RUN_BIN=foo run`
CUDA_ID ?= 1

# Default binary to run with `make run` (can be overridden)
RUN_BIN ?= mma_sp_m16n8k32_fp32fp16

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

# Map numeric RUN_BIN to binary name (1=first, 2=second, etc.)
BIN_TO_RUN := $(strip $(if $(filter $(RUN_BIN),$(BINS)), $(RUN_BIN), $(word $(RUN_BIN),$(BINS))))

.PHONY: all clean run run-all info list
all: $(BINS)

# Pattern rule: compile %.cu -> %
%: %.cu
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $<

# FP8 kernels require sm_89+ (Hopper); build them with sm_90.
mma_sp_%fp8: ARCH := $(ARCH_FP8)

# Default run: runs the configured RUN_BIN (must exist)
run: $(BIN_TO_RUN)
	@echo "Running $(BIN_TO_RUN) on CUDA device $(CUDA_ID)..." \
	&& CUDA_VISIBLE_DEVICES=$(CUDA_ID) ./$(BIN_TO_RUN)

# List available run targets
list:
	@echo "Available binaries:"; printf "  %s\n" $(BINS)
	@echo "Run a binary: make RUN_BIN=<name> run  (or RUN_BIN=1 for first, 2 for second, etc.)"

# Print discovery / config info
info: ; @echo "SRCS = $(SRCS)"
	@echo "BINS = $(BINS)"
	@echo "RUN_BIN = $(RUN_BIN) (BIN_TO_RUN = $(BIN_TO_RUN))"
	@echo "CUDA_ID = $(CUDA_ID) (override with 'make CUDA_ID=0')"

clean:
	rm -f $(BINS)
