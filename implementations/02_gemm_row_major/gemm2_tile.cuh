#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#define Ms 32
#define Ns 32
#define Ks 32

// (M, K) x (K, N) = (M, N)
__global__ void sgemm2(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb,
                      float beta, float *C, int ldc) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    A = &(A[bx * Ms * lda]);
    B = &(B[by * Ns]);
    C = &(C[bx * Ms * ldc + by * Ns]);

    __shared__ float a_shared[Ms][Ks];
    __shared__ float b_shared[Ks][Ns];

    float sum = 0.f;
    for (int k_idx = 0; k_idx < K / Ks; ++k_idx) {
        a_shared[ty][tx] = A[ty * lda + tx];
        b_shared[ty][tx] = B[ty * ldb + tx];
        A = A + Ks;
        B = B + Ks * ldb;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < Ks; ++kk) {
            sum += a_shared[kk][tx] * b_shared[ty][kk];
        }

    }
    sum = sum * alpha + beta * C[tx * ldc + ty];
    C[tx * ldc + ty] = sum;
}

void test_sgemm2(int M, int N, int K, float *alpha, float *dA, int lda, float *dB, int ldb, 
                float *beta, float *dC, int ldc) {
    dim3 block(Ns, Ms);
    dim3 grid((N + Ns - 1) / Ns, (M + Ms - 1) / Ms);
    
    sgemm2<<<grid, block>>>(M, N, K, *alpha, dA, lda, dB, ldb, *beta, dC, ldc);
    cudaDeviceSynchronize();
    
}