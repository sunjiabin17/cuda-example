#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#define Ms64 64
#define Ns64 64
#define Ks16 16

#define Mr4 4
#define Nr4 4

// 一个线程计算4x4
// ({Ms64,Ns64,Ks16}={64,64,16}, {Mr4,Nr4}={4,4})
__global__ void sgemm4(int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb,
                      float beta, float *C, int ldc) {
    int tx = threadIdx.x, ty = threadIdx.y; // 线程索引
    int bx = blockIdx.x, by = blockIdx.y;   // 线程块索引

    // int col_a = (tx&3) << 2;
    // int row_a = (ty<<2) + (tx>>2);

    int row_a = tx << 2;
    int col_a = ty;

    int row_b = tx;
    int col_b = ty << 2;

    int row_c = tx << 2;
    int col_c = ty << 2;


    float* pA = &(A[bx * Ms64 * lda]);
    float* pB = &(B[by * Ns64]);
    float* pC = &(C[bx * Ms64 * ldc + by * Ns64]);

    __shared__ float a_shared[Ms64][Ks16];
    __shared__ float b_shared[Ks16][Ns64];

    float4 Av, Bv, Cv[4], sums[4];
    memset(sums, 0.f, sizeof(sums));

    for (int k_idx = 0; k_idx < K; k_idx += Ks16) {
        // Av = *((float4 *)(&pA[row_a * lda + col_a]));
        // Bv = *((float4 *)(&pB[row_b * ldb + col_b]));

        // ((float4 *)a_shared)[block_y*Ks16 + block_x] = Av;
        // a_shared[col_a][row_a] = Av.x;
        // a_shared[col_a+1][row_a] = Av.y;
        // a_shared[col_a+2][row_a] = Av.z;
        // a_shared[col_a+3][row_a] = Av.w;
        // ((float4 *)b_shared)[row_b*Ns64 + col_b] = Bv;

        a_shared[row_a][col_a] = pA[row_a*lda + col_a];
        a_shared[row_a+1][col_a] = pA[(row_a+1)*lda + col_a];
        a_shared[row_a+2][col_a] = pA[(row_a+2)*lda + col_a];
        a_shared[row_a+3][col_a] = pA[(row_a+3)*lda + col_a];

        b_shared[row_b][col_b] = pB[row_b*ldb + col_b];
        b_shared[row_b][col_b+1] = pB[row_b*ldb + col_b+1];
        b_shared[row_b][col_b+2] = pB[row_b*ldb + col_b+2];
        b_shared[row_b][col_b+3] = pB[row_b*ldb + col_b+3];

        pA = pA + Ks16;
        pB = pB + Ks16 * ldb;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < Ks16; ++kk) {
            // Av = *((float4 *)(&a_shared[kk][row_c])); // A中的4个数分别对应C的每一行
            // Bv = *((float4 *)(&b_shared[kk][col_c])); // B中的4个数分别对应C的每一列
            sums[0].x += a_shared[row_a][kk] * b_shared[kk][col_b];
            sums[0].y += a_shared[row_a][kk] * b_shared[kk][col_b+1];
            sums[0].z += a_shared[row_a][kk] * b_shared[kk][col_b+2];
            sums[0].w += a_shared[row_a][kk] * b_shared[kk][col_b+3];

            sums[1].x += a_shared[row_a+1][kk] * b_shared[kk][col_b];
            sums[1].y += a_shared[row_a+1][kk] * b_shared[kk][col_b+1];
            sums[1].z += a_shared[row_a+1][kk] * b_shared[kk][col_b+2];
            sums[1].w += a_shared[row_a+1][kk] * b_shared[kk][col_b+3];

            sums[2].x += a_shared[row_a+2][kk] * b_shared[kk][col_b];
            sums[2].y += a_shared[row_a+2][kk] * b_shared[kk][col_b+1];
            sums[2].z += a_shared[row_a+2][kk] * b_shared[kk][col_b+2];
            sums[2].w += a_shared[row_a+2][kk] * b_shared[kk][col_b+3];

            sums[3].x += a_shared[row_a+3][kk] * b_shared[kk][col_b];
            sums[3].y += a_shared[row_a+3][kk] * b_shared[kk][col_b+1];
            sums[3].z += a_shared[row_a+3][kk] * b_shared[kk][col_b+2];
            sums[3].w += a_shared[row_a+3][kk] * b_shared[kk][col_b+3];


            // C的一个block中的一行, 内存连续, 对应sums[0]
            // sums[0].x += Av.x * Bv.x;
            // sums[0].y += Av.x * Bv.y;
            // sums[0].z += Av.x * Bv.z;
            // sums[0].w += Av.x * Bv.w;
            
            // sums[1].x += Av.y * Bv.x;
            // sums[1].y += Av.y * Bv.y;
            // sums[1].z += Av.y * Bv.z;
            // sums[1].w += Av.y * Bv.w;

            // sums[2].x += Av.z * Bv.x;
            // sums[2].y += Av.z * Bv.y;
            // sums[2].z += Av.z * Bv.z;
            // sums[2].w += Av.z * Bv.w;

            // sums[3].x += Av.w * Bv.x;
            // sums[3].y += Av.w * Bv.y;
            // sums[3].z += Av.w * Bv.z;
            // sums[3].w += Av.w * Bv.w;
        }
        __syncthreads();
    }
    // Cv[0] = *((float4 *)(&pC[row_c*ldc + col_c]));
    // Cv[1] = *((float4 *)(&pC[(row_c+1)*ldc + col_c]));
    // Cv[2] = *((float4 *)(&pC[(row_c+2)*ldc + col_c]));
    // Cv[3] = *((float4 *)(&pC[(row_c+3)*ldc + col_c]));

    // sums[0].x = alpha * sums[0].x + beta * Cv[0].x;

    *((float4 *)(&pC[row_c*ldc + col_c])) = sums[0];
    *((float4 *)(&pC[(row_c+1)*ldc + col_c])) = sums[1];
    *((float4 *)(&pC[(row_c+2)*ldc + col_c])) = sums[2];
    *((float4 *)(&pC[(row_c+3)*ldc + col_c])) = sums[3];
}


void test_sgemm4(int M, int N, int K, float *alpha, float *dA, int lda, float *dB, int ldb, 
                float *beta, float *dC, int ldc) {
    dim3 block(16, 16);
    dim3 grid((M + Ms64 - 1) / Ms64, (N + Ns64 - 1) / Ns64);
    
    sgemm4<<<grid, block>>>(M, N, K, *alpha, dA, lda, dB, ldb, *beta, dC, ldc);
    cudaDeviceSynchronize();
    
}