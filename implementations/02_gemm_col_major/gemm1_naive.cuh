#define bM 32
#define bN 32
#define bK 32

__global__ __launch_bounds__(1024) void gemm1_naive(int M, int N, int K, float* A, float* B, float* C) {
    // 列优先
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    float tmp = 0.0f;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);
    for (int k_idx = 0; k_idx < K; k_idx++) {
        tmp += A[tx + k_idx * lda] * B[k_idx + ty * ldb];
    }
    C[tx + ty * ldc] = tmp;
}

