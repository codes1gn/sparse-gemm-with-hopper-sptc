
#include <cuda_runtime.h>
#include <iostream>

__global__ void test_tf32_sparse() {
    float c[4] = {0,0,0,0};
    float a[4] = {1,1,1,1};
    float b[4] = {1,1,1,1};
    int e = 0;
    
    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(*(int*)&a[0]), "r"(*(int*)&a[1]), "r"(*(int*)&a[2]), "r"(*(int*)&a[3]),
          "r"(*(int*)&b[0]), "r"(*(int*)&b[1]), "r"(*(int*)&b[2]), "r"(*(int*)&b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(e)
    );
}

int main() {
    test_tf32_sparse<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
