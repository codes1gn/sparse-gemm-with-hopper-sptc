
NVCC = /usr/local/cuda/bin/nvcc
ARCH = -arch=sm_80
FLAGS = -O3 -std=c++14

all: sparse_gemm

sparse_gemm: sparse_gemm.cu
	$(NVCC) $(ARCH) $(FLAGS) -o sparse_gemm sparse_gemm.cu

run: sparse_gemm
	./sparse_gemm

clean:
	rm -f sparse_gemm
