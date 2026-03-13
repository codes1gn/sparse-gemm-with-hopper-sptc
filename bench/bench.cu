#include <iostream>
#include <string>
#include <random>
#include <vector>
#include "bench_common.hpp"
#include "bench_fp16.cuh"
#include "bench_e4m3.cuh"

int main(int argc, char** argv) {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    std::cout << "========================================" << std::endl;
    std::cout << "Structured Sparsity GEMM Benchmark" << std::endl;
    std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n--- FP16 Benchmarks ---" << std::endl;
    bool fp16_ok = run_fp16_bench_suite(M, N, K);

    std::cout << "\n--- FP8 e4m3 Benchmarks ---" << std::endl;
    bool e4m3_ok = run_e4m3_bench_suite(M, N, K);

    std::cout << "\n========================================" << std::endl;
    if (fp16_ok && e4m3_ok) {
        std::cout << "All benchmarks PASSED" << std::endl;
    } else {
        std::cout << "Some benchmarks FAILED" << std::endl;
        return 1;
    }
    std::cout << "========================================" << std::endl;

    return 0;
}
