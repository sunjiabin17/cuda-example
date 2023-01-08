#define bM 32
#define bN 32
#define bK 32

__global__ __launch_bounds__(1024) void gemm5_reduce_bank_conflict(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim block
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);

    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmp = 0.0f;
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[row + col * bK] = A[row + col * lda];    // update
        b_tile[row + col * bK] = B[row + col * ldb];    // update b_tile按row连续
        A += (lda << 5);
        B += bK;
        __syncthreads();
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            // memory coelascing on shared memory
            tmp += a_tile[row + inner_k * bK] * b_tile[inner_k + col * bK]; // update
        }
        __syncthreads();
    }
    C[row + col * ldc] = tmp;
}

