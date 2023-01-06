#define bM 32
#define bN 32
#define bK 32

#define IDX2C(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX(i,j) IDX2C(i,j,bK)         // shared memory index

__global__ __launch_bounds__(256) void gemm7_4x1_vectorized_load_store(int M, int N, int K, float* A, float* B, float* C) {
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

    float4 Av, Bv, Cv, tmps;
    tmps.x = 0.0f, tmps.y = 0.0f, tmps.z = 0.0f, tmps.w = 0.0f;
    
    float b00;
    
    for (int k_idx = 0; k_idx < K; k_idx += bK) {
        Av = *((float4*)&(A[IDX2C(row1, col, lda)]));
        Bv = *((float4*)&(B[IDX2C(row1, col, ldb)]));
        ((float4*)a_tile)[tx] = Av;
        ((float4*)b_tile)[tx] = Bv;     // 耗时更少
        // b_tile[SIDX(col, row1)] = Bv.x;
        // b_tile[SIDX(col, row2)] = Bv.y;
        // b_tile[SIDX(col, row3)] = Bv.z;
        // b_tile[SIDX(col, row4)] = Bv.w;
        A += (lda << 5);
        B += bK;
        __syncthreads();
        #pragma unroll
        for (int inner_k = 0; inner_k < bK; ++inner_k) {
            // b00 = b_tile[SIDX(col, inner_k)];
            b00 = b_tile[SIDX(inner_k, col)];   // 耗时更少
            tmps.x += a_tile[SIDX(row1, inner_k)] * b00;
            tmps.y += a_tile[SIDX(row2, inner_k)] * b00;
            tmps.z += a_tile[SIDX(row3, inner_k)] * b00;
            tmps.w += a_tile[SIDX(row4, inner_k)] * b00;
        }
        __syncthreads();
    }
    Cv = *((float4*)(&C[IDX2C(row1, col, ldc)]));   // C在(row1, col)位置的地址强转成float4*类型，然后取值给Cv
    Cv.x = tmps.x;
    Cv.y = tmps.y;
    Cv.z = tmps.z;
    Cv.w = tmps.w;
    *((float4*)(&C[IDX2C(row1, col, ldc)])) = Cv; 
}

