#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#define Ms 32
#define Ns 32
#define Ks 32

// 一个线程计算4x1
// (M, K) x (K, N) = (M, N)
__global__ void sgemm3(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb,
                      float beta, float *C, int ldc) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int sx = tx<<1, sy = ty<<1;

    float* pA = &(A[bx * Ms * lda]);
    float* pB = &(B[by * Ns]);
    float* pC = &(C[bx * Ms * ldc + by * Ns]);

    __shared__ float a_shared[Ms][Ks];
    __shared__ float b_shared[Ks][Ns];

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k_idx = 0; k_idx < K; k_idx += Ks) {
        a_shared[sy][sx] = pA[sy * lda + sx];
        a_shared[sy][sx+1] = pA[sy * lda + sx+1];
        a_shared[sy+1][sx] = pA[(sy+1) * lda + sx];
        a_shared[sy+1][sx+1] = pA[(sy+1) * lda + sx+1];
        b_shared[sy][sx] = pB[sy * ldb + sx];
        b_shared[sy][sx+1] = pB[sy * ldb + sx+1];
        b_shared[sy+1][sx] = pB[(sy+1) * ldb + sx];
        b_shared[sy+1][sx+1] = pB[(sy+1) * ldb + sx+1];

        pA = pA + Ks;
        pB = pB + Ks * ldb;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < Ks; ++kk) {
            sums[0] += a_shared[tx][kk] * b_shared[kk][ty];
            sums[1] += a_shared[tx][kk] * b_shared[kk][ty+1];
            sums[2] += a_shared[tx+1][kk] * b_shared[kk][ty];
            sums[3] += a_shared[tx+1][kk] * b_shared[kk][ty+1];
        }
        __syncthreads();
    }
    // sum = sum * alpha + beta * C[tx * ldc + ty];

    pC[tx * ldc + ty] = sums[0];
    pC[tx * ldc + ty+1] = sums[1];
    pC[(tx+1) * ldc + ty] = sums[2];
    pC[(tx+1) * ldc + ty+1] = sums[3];
}


void test_sgemm3(int M, int N, int K, float *alpha, float *dA, int lda, float *dB, int ldb, 
                float *beta, float *dC, int ldc) {
    dim3 block(16, 16);
    dim3 grid((M + Ms - 1) / Ms, (N + Ns - 1) / Ns);
    
    sgemm3<<<grid, block>>>(M, N, K, *alpha, dA, lda, dB, ldb, *beta, dC, ldc);
    cudaDeviceSynchronize();
    
}