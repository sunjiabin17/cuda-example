#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#define Ms 32
#define Ns 32
#define Ks 32

// 一个线程计算2x2
// (M, K) x (K, N) = (M, N)
__global__ void sgemm3(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb,
                      float beta, float *C, int ldc) {
    int tx = threadIdx.x, ty = threadIdx.y; // 线程索引
    int bx = blockIdx.x, by = blockIdx.y;   // 线程块索引

    int block_x = tx<<1, block_y = ty<<1;   // 对应的矩阵块中的索引

    float* pA = &(A[bx * Ms * lda]);
    float* pB = &(B[by * Ns]);
    float* pC = &(C[bx * Ms * ldc + by * Ns]);

    __shared__ float a_shared[Ms][Ks];
    __shared__ float b_shared[Ks][Ns];

    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k_idx = 0; k_idx < K; k_idx += Ks) {
        a_shared[block_y][block_x] = pA[block_y * lda + block_x];
        a_shared[block_y][block_x+1] = pA[block_y * lda + block_x+1];
        a_shared[block_y+1][block_x] = pA[(block_y+1) * lda + block_x];
        a_shared[block_y+1][block_x+1] = pA[(block_y+1) * lda + block_x+1];
        b_shared[block_y][block_x] = pB[block_y * ldb + block_x];
        b_shared[block_y][block_x+1] = pB[block_y * ldb + block_x+1];
        b_shared[block_y+1][block_x] = pB[(block_y+1) * ldb + block_x];
        b_shared[block_y+1][block_x+1] = pB[(block_y+1) * ldb + block_x+1];

        pA = pA + Ks;
        pB = pB + Ks * ldb;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < Ks; ++kk) {
            sums[0] += a_shared[block_x][kk] * b_shared[kk][block_y];
            sums[1] += a_shared[block_x][kk] * b_shared[kk][block_y+1];
            sums[2] += a_shared[block_x+1][kk] * b_shared[kk][block_y];
            sums[3] += a_shared[block_x+1][kk] * b_shared[kk][block_y+1];
        }
        __syncthreads();
    }
    // sum = sum * alpha + beta * C[block_x * ldc + block_y];

    pC[block_x * ldc + block_y] = sums[0];
    pC[block_x * ldc + block_y+1] = sums[1];
    pC[(block_x+1) * ldc + block_y] = sums[2];
    pC[(block_x+1) * ldc + block_y+1] = sums[3];
}


void test_sgemm3(int M, int N, int K, float *alpha, float *dA, int lda, float *dB, int ldb, 
                float *beta, float *dC, int ldc) {
    dim3 block(16, 16);
    dim3 grid((M + Ms - 1) / Ms, (N + Ns - 1) / Ns);
    
    sgemm3<<<grid, block>>>(M, N, K, *alpha, dA, lda, dB, ldb, *beta, dC, ldc);
    cudaDeviceSynchronize();
    
}