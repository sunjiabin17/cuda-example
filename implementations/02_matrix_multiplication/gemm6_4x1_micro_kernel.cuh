#define bM 32
#define bN 32
#define bK 32

#define IDX2C(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX(i,j) IDX2C(i,j,bK)         // shared memory index

__global__ __launch_bounds__(256) void gemm6_4x1_micro_kernel(int M, int N, int K, float* A, float* B, float* C) {
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim block 16*16
    int bx = blockIdx.x, by = blockIdx.y;
    // 4个thread处理同一行(一列)数据
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3;
    int col = tx>>3;
    A = &(A[(bx << 5)]);
    B = &(B[(by << 5) * ldb]);
    C = &(C[(bx << 5) + (by << 5) * ldc]);

    __shared__ float a_tile[bM*bK];
    __shared__ float b_tile[bK*bN];

    float tmps[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b00;
    
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        a_tile[SIDX(row1, col)] = A[IDX2C(row1, col, lda)];
        a_tile[SIDX(row2, col)] = A[IDX2C(row2, col, lda)];
        a_tile[SIDX(row3, col)] = A[IDX2C(row3, col, lda)];
        a_tile[SIDX(row4, col)] = A[IDX2C(row4, col, lda)];
        // b_tile[SIDX(col, row1)] = B[IDX2C(row1, col, ldb)];
        // b_tile[SIDX(col, row2)] = B[IDX2C(row2, col, ldb)];
        // b_tile[SIDX(col, row3)] = B[IDX2C(row3, col, ldb)];
        // b_tile[SIDX(col, row4)] = B[IDX2C(row4, col, ldb)];
        b_tile[SIDX(row1, col)] = B[IDX2C(row1, col, ldb)];  // 耗时更少
        b_tile[SIDX(row2, col)] = B[IDX2C(row2, col, ldb)];
        b_tile[SIDX(row3, col)] = B[IDX2C(row3, col, ldb)];
        b_tile[SIDX(row4, col)] = B[IDX2C(row4, col, ldb)];
        A += (lda << 5);
        B += bK;
        __syncthreads();
        #pragma unroll
        for (int inner_k = 0; inner_k < bK; inner_k++) {
            // b00 = b_tile[SIDX(col, inner_k)];
            b00 = b_tile[SIDX(inner_k, col)];    // 耗时更少
            tmps[0] += a_tile[SIDX(row1, inner_k)] * b00;
            tmps[1] += a_tile[SIDX(row2, inner_k)] * b00;
            tmps[2] += a_tile[SIDX(row3, inner_k)] * b00;
            tmps[3] += a_tile[SIDX(row4, inner_k)] * b00;
        }
        __syncthreads();
    }

    C[IDX2C(row1, col, ldc)] = tmps[0];
    C[IDX2C(row2, col, ldc)] = tmps[1];
    C[IDX2C(row3, col, ldc)] = tmps[2];
    C[IDX2C(row4, col, ldc)] = tmps[3];
}

