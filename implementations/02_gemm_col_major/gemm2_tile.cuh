#define bM 32
#define bN 32
#define bK 32

__global__ __launch_bounds__(1024) void gemm2_tile(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);
    // A = &A((bx<<5),0);
    // B = &B(0,(by<<5));
    // C = &C((bx<<5),(by<<5));
    
    // update: tile share memory
    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmp = 0.0f;
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[tx * bK + ty] = A[tx + ty * lda]; // bank conflict
        b_tile[ty * bK + tx] = B[tx + ty * ldb];
        A += (lda << 5);
        B += bK;
        __syncthreads();
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            tmp += a_tile[tx * bK + inner_k] * b_tile[ty * bK + inner_k];
        }
        __syncthreads(); 
    }
    C[tx + ty * ldc] = 1.f * tmp + 0.f * C[tx + ty * ldc];

}

