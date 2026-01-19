
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch=sm_80
FLAGS = -O3 -std=c++14

# Default CUDA device id (can be overridden on the make command line)
# Example: `make CUDA_ID=0 run-foo` or `make CUDA_ID=0 RUN_BIN=foo run`
CUDA_ID ?= 1

# Default binary to run with `make run` (can be overridden)
RUN_BIN ?= mma_sp_m16n8k32_fp32fp16

# Discover only mma_sp_* CUDA sources so each kernel gets its own binary
SRCS := $(wildcard mma_sp_*.cu)
BINS := $(SRCS:.cu=)

# Map numeric RUN_BIN to binary name (1=first, 2=second, etc.)
BIN_TO_RUN := $(strip $(if $(filter 1 2, $(RUN_BIN)), $(word $(RUN_BIN), $(BINS)), $(RUN_BIN)))

.PHONY: all clean run run-all info list
all: $(BINS)

# Pattern rule: compile %.cu -> %
%: %.cu
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $<

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
