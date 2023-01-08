#include <cublas_v2.h>
#include <cuda_runtime.h>

// (M, K) x (K, N) = (M, N)
__global__ void sgemm1(int M, int N, int K, float *A, int lda, float *B, int ldb,
                      float *C, int ldc) {
    int lda = K, ldb = N, ldc = N;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.f;
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        // k_idx每次循环, 从A矩阵取一行元素, 从B矩阵取一列元素
        sum += A[x * lda + k_idx] * B[k_idx * ldb + y];
    }
    C[x * ldc + y] = sum;
}
