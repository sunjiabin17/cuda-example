#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>


// (M, K) x (K, N) = (M, N)
__global__ void sgemm1(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb,
                      float beta, float *C, int ldc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.f;
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        // k_idx每次循环, 从A矩阵取一行元素, 从B矩阵取一列元素
        sum += A[x * lda + k_idx] * B[k_idx * ldb + y];
    }
    C[x * ldc + y] = alpha * C[x * ldc + y] + beta * sum;
}

void test_sgemm1(int M, int N, int K, float *alpha, float *dA, int lda, float *dB, int ldb, 
                float *beta, float *dC, int ldc) {
    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    sgemm1<<<grid, block>>>(M, N, K, *alpha, dA, lda, dB, ldb, *beta, dC, ldc);
    cudaDeviceSynchronize();
    
}