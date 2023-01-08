//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;

#define vfma(v1, alpha, v2, beta, v3)\
    v1.x = alpha * v2.x + beta * v3.x;\
    v1.y = alpha * v2.y + beta * v3.y;\
    v1.z = alpha * v2.z + beta * v3.z;\
    v1.w = alpha * v2.w + beta * v3.w;

#define bM10 128
#define bN10 128
#define bK10 8

#define IDX10(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX10(i,j) IDX10(i,j,(1<<7))     // shared memory index

// ({Ms,Ns,Ks}={128,128,8}, {Mr,Nr}={8,8}) {Mw,Nw}={4xMr,8xNr}
__global__ __launch_bounds__(256) void gemm10_8x8_warp_tile(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 128*128, thread size: 16*16, 每个block有8个warp
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;           // tx = 0-255
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx >> 5;              // tx/32, warp_id = 0-7
    int lane_id = tx & 31;              // tx%32, lane_id = 0-31
    int warp_row = warp_id & 3;         // warp_id%4, warp_row = 0-3
    int warp_col = warp_id >> 2;        // warp_id/4, warp_col = 0-1
    int row_w = lane_id & 3;            // lane_id%4, row_w = 0-3
    int col_w = lane_id >> 2;           // lane_id/4, col_w = 0-7

    int row_a = (tx&31)<<2, col_a = tx>>5;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    

    int row_c = (warp_row<<5) + (row_w<<3); // stride=8
    int col_c = (warp_col<<6) + (col_w<<3);
    // threadIdx.x  : 0, 1, 2, 3, 4, 5, 6, 7  8, 9,10,11,12,13,14,15   16,17,18,19,20,21,22,23  24,25,26,27,28,29,30,31
    // row_c        : 0, 8,16,24, 0, 8,16,24  0, 8,16,24, 0, 8,16,24    0, 8,16,24, 0, 8,16,24   0, 8,16,24, 0, 8,16,24
    // col_c        : 0, 0, 0, 0, 8, 8, 8, 8  16,16,16,16,24,24,24,24  32,32,32,32,40,40,40,40  48,48,48,48,56,56,56,56
    // 同一个warp中, shared memory A地址的起始地址0, 终止地址24+8=32, 32/32=4, 需要1次transaction
    //             shared memory B地址的起始地址0, 终止地址56+8=64, 64/32=2, 需要2次transaction

    // gemm9
    // int row_a = (tx&31)<<2, col_a = tx>>5;  // row_a = range(0, 128, 4), col_a = range(0, 8, 1)
    // int row_b = (tx&1)<<2, col_b = tx>>1;   // row_b = range(0, 8, 4), col_b = range(0, 128, 1)
    // int row_c = (tx&15)<<3, col_c = (tx>>4)<<3; // row_c = range(0, 128, 8), col_c = range(0, 128, 8)
    // threadIdx.x  : 0, 1, 2, 3, 4, 5, 6, 7  8, 9,10,11,12, 13, 14, 15   16,17,18,19,20,21,22,23  24,25,26,27,28, 29, 30, 31
    //  row_c       : 0, 8,16,24,32,40,48,56  64,72,80,88,96,104,112,120   0, 8,16,24,32,40,48,56  64,72,80,88,96,104,112,120
    //  col_c       : 0, 0, 0, 0, 0, 0, 0, 0  8, 8, 8, 8, 8,  8,  8,  8   16,16,16,16,16,16,16,16  24,24,24,24,24, 24, 24, 24
    // 同一个warp中, shared memory A地址的起始地址0, 终止地址120+8=128, 128/32=4, 需要4次transaction
    //              shared memory B地址的起始地址0, 终止地址24+8=32, 32/32=1， 需要1次transaction
    A = &(A[(bx<<7)]);
    B = &(B[(by<<7) * ldb]);
    C = &(C[(bx<<7) + (by<<7) * ldc]);

    __shared__ float a_tile[bM10*bK10];
    __shared__ float b_tile[bK10*bN10];

    float4 Av1, Av2, Bv1, Bv2, Cv[16], tmps[16];
    memset(tmps, 0, sizeof(tmps));

    for (int k_idx = 0; k_idx < K; k_idx += bK10) {
        Av1 = *((float4 *)(&A[IDX10(row_a, col_a, lda)]));
        Bv1 = *((float4 *)(&B[IDX10(row_b, col_b, ldb)]));
        ((float4 *)a_tile)[tx] = Av1;
        b_tile[SIDX10(col_b, row_b)] = Bv1.x;
        b_tile[SIDX10(col_b, row_b+1)] = Bv1.y;
        b_tile[SIDX10(col_b, row_b+2)] = Bv1.z;
        b_tile[SIDX10(col_b, row_b+3)] = Bv1.w;
        A += lda8; B += bK10;
        __syncthreads();

        #pragma unroll
        for (int inner_k = 0; inner_k < bK10; ++inner_k) {
            Av1 = *((float4 *)(&a_tile[SIDX10(row_c, inner_k)]));
            Av2 = *((float4 *)(&a_tile[SIDX10(row_c+4, inner_k)]));
            Bv1 = *((float4 *)(&b_tile[SIDX10(col_c, inner_k)]));
            Bv2 = *((float4 *)(&b_tile[SIDX10(col_c+4, inner_k)]));
            vscal(tmps[0], Av1, Bv1.x)     // tmps[x][y] 对应计算A[x][:] * B[:][y], 结果存在C[x][y]处
            vscal(tmps[1], Av2, Bv1.x)
            vscal(tmps[2], Av1, Bv1.y)
            vscal(tmps[3], Av2, Bv1.y)
            vscal(tmps[4], Av1, Bv1.z)
            vscal(tmps[5], Av2, Bv1.z)
            vscal(tmps[6], Av1, Bv1.w)
            vscal(tmps[7], Av2, Bv1.w)
            vscal(tmps[8], Av1, Bv2.x)
            vscal(tmps[9], Av2, Bv2.x)
            vscal(tmps[10], Av1, Bv2.y)
            vscal(tmps[11], Av2, Bv2.y)
            vscal(tmps[12], Av1, Bv2.z)
            vscal(tmps[13], Av2, Bv2.z)
            vscal(tmps[14], Av1, Bv2.w)
            vscal(tmps[15], Av2, Bv2.w)
        }
        __syncthreads();
    }

    // 每个线程从全局内存中读取8*8个数
    // 从全局内存中取出row_a, col_c处连续的4个值给Cv
    Cv[0] = *((float4 *)(&C[IDX10(row_c, col_c, ldc)]));
    Cv[1] = *((float4 *)(&C[IDX10(row_c+4, col_c, ldc)]));
    Cv[2] = *((float4 *)(&C[IDX10(row_c, col_c+1, ldc)]));
    Cv[3] = *((float4 *)(&C[IDX10(row_c+4, col_c+1, ldc)]));
    Cv[4] = *((float4 *)(&C[IDX10(row_c, col_c+2, ldc)]));
    Cv[5] = *((float4 *)(&C[IDX10(row_c+4, col_c+2, ldc)]));
    Cv[6] = *((float4 *)(&C[IDX10(row_c, col_c+3, ldc)]));
    Cv[7] = *((float4 *)(&C[IDX10(row_c+4, col_c+3, ldc)]));
    Cv[8] = *((float4 *)(&C[IDX10(row_c, col_c+4, ldc)]));
    Cv[9] = *((float4 *)(&C[IDX10(row_c+4, col_c+4, ldc)]));
    Cv[10] = *((float4 *)(&C[IDX10(row_c, col_c+5, ldc)]));
    Cv[11] = *((float4 *)(&C[IDX10(row_c+4, col_c+5, ldc)]));
    Cv[12] = *((float4 *)(&C[IDX10(row_c, col_c+6, ldc)]));
    Cv[13] = *((float4 *)(&C[IDX10(row_c+4, col_c+6, ldc)]));
    Cv[14] = *((float4 *)(&C[IDX10(row_c, col_c+7, ldc)]));
    Cv[15] = *((float4 *)(&C[IDX10(row_c+4, col_c+7, ldc)]));

    // 每个线程进行8*8次计算
    vfma(tmps[0], alpha, tmps[0], beta, Cv[0])
    vfma(tmps[1], alpha, tmps[1], beta, Cv[1])
    vfma(tmps[2], alpha, tmps[2], beta, Cv[2])
    vfma(tmps[3], alpha, tmps[3], beta, Cv[3])
    vfma(tmps[4], alpha, tmps[4], beta, Cv[4])
    vfma(tmps[5], alpha, tmps[5], beta, Cv[5])
    vfma(tmps[6], alpha, tmps[6], beta, Cv[6])
    vfma(tmps[7], alpha, tmps[7], beta, Cv[7])
    vfma(tmps[8], alpha, tmps[8], beta, Cv[8])
    vfma(tmps[9], alpha, tmps[9], beta, Cv[9])
    vfma(tmps[10], alpha, tmps[10], beta, Cv[10])
    vfma(tmps[11], alpha, tmps[11], beta, Cv[11])
    vfma(tmps[12], alpha, tmps[12], beta, Cv[12])
    vfma(tmps[13], alpha, tmps[13], beta, Cv[13])
    vfma(tmps[14], alpha, tmps[14], beta, Cv[14])
    vfma(tmps[15], alpha, tmps[15], beta, Cv[15])

    // 每个线程写入8*8个数到全局内存
    *((float4 *)(&C[IDX10(row_c, col_c, ldc)])) = tmps[0];
    *((float4 *)(&C[IDX10(row_c+4, col_c, ldc)])) = tmps[1];
    *((float4 *)(&C[IDX10(row_c, col_c+1, ldc)])) = tmps[2];
    *((float4 *)(&C[IDX10(row_c+4, col_c+1, ldc)])) = tmps[3];
    *((float4 *)(&C[IDX10(row_c, col_c+2, ldc)])) = tmps[4];
    *((float4 *)(&C[IDX10(row_c+4, col_c+2, ldc)])) = tmps[5];
    *((float4 *)(&C[IDX10(row_c, col_c+3, ldc)])) = tmps[6];
    *((float4 *)(&C[IDX10(row_c+4, col_c+3, ldc)])) = tmps[7];
    *((float4 *)(&C[IDX10(row_c, col_c+4, ldc)])) = tmps[8];
    *((float4 *)(&C[IDX10(row_c+4, col_c+4, ldc)])) = tmps[9];
    *((float4 *)(&C[IDX10(row_c, col_c+5, ldc)])) = tmps[10];
    *((float4 *)(&C[IDX10(row_c+4, col_c+5, ldc)])) = tmps[11];
    *((float4 *)(&C[IDX10(row_c, col_c+6, ldc)])) = tmps[12];
    *((float4 *)(&C[IDX10(row_c+4, col_c+6, ldc)])) = tmps[13];
    *((float4 *)(&C[IDX10(row_c, col_c+7, ldc)])) = tmps[14];
    *((float4 *)(&C[IDX10(row_c+4, col_c+7, ldc)])) = tmps[15];
}