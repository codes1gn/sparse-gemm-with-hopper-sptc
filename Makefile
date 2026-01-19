
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch=sm_80
FLAGS = -O3 -std=c++14

# Default CUDA device id (can be overridden on the make command line)
# Example: `make CUDA_ID=0 run-foo` or `make CUDA_ID=0 RUN_BIN=foo run`
CUDA_ID ?= 1

# Default binary to run with `make run` (can be overridden)
RUN_BIN ?= sparse_gemm

# Discover all .cu sources in the repository root and create same-named binaries
SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

.PHONY: all clean run run-all info list $(patsubst %,%,$(addprefix run-,${BINS}))

all: $(BINS)

# Pattern rule: compile %.cu -> %
%: %.cu
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $<

# Default run: runs the configured RUN_BIN (must exist)
run: $(RUN_BIN)
	@echo "Running $(RUN_BIN) on CUDA device $(CUDA_ID)..." \
	&& CUDA_VISIBLE_DEVICES=$(CUDA_ID) ./$(RUN_BIN)

# Per-binary run targets: `make run-<name>`
run-%: %
	@echo "Running $* on CUDA device $(CUDA_ID)..." \
	&& CUDA_VISIBLE_DEVICES=$(CUDA_ID) ./$(patsubst run-%,%,$@)

# Run all built binaries sequentially (uses same CUDA_ID for each)
run-all: $(BINS)
	@for b in $(BINS); do \
		echo "---- running $$b (CUDA_VISIBLE_DEVICES=$(CUDA_ID))"; \
		CUDA_VISIBLE_DEVICES=$(CUDA_ID) ./$$b || exit 1; \
	done

# Print discovery / config info
info:
	@echo "SRCS = $(SRCS)"
	@echo "BINS = $(BINS)"
	@echo "RUN_BIN = $(RUN_BIN) (override with 'make RUN_BIN=name')"
	@echo "CUDA_ID = $(CUDA_ID) (override with 'make CUDA_ID=0')"

# List available run targets
list:
	@echo "Available binaries:"; printf "  %s\n" $(BINS)
	@echo "Run a binary: make run-<name>   (or: make RUN_BIN=<name> run)"

clean:
	rm -f $(BINS)
