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

#define bM9 128
#define bN9 128
#define bK9 8

#define IDX9(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX9(i,j) IDX9(i,j,(1<<7))     // shared memory index

// ({Ms,Ns,Ks}={128,128,8}, {Mr,Nr}={8,8})
__global__ __launch_bounds__(256) void gemm9_8x8_micro_kernel(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 128*128, thread size: 16*16
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim
    int bx = blockIdx.x, by = blockIdx.y;
    

    int row_a = (tx&31)<<2, col_a = tx>>5;  // row_a = range(0, 128, 4), col_a = range(0, 8, 1)
    int row_b = (tx&1)<<2, col_b = tx>>1;   // row_b = range(0, 8, 4), col_b = range(0, 128, 1)
    int row_c = (tx&15)<<3, col_c = (tx>>4)<<3; // row_c = range(0, 128, 8), col_c = range(0, 128, 8)
    int lda8 = lda<<3;

    A = &(A[(bx << 7)]);
    B = &(B[(by << 7) * ldb]);
    C = &(C[(bx << 7) + (by << 7) * ldc]);  // the TB size is 128.

    __shared__ float a_tile[bM9*bK9];
    __shared__ float b_tile[bK9*bN9];

    float4 Av1, Av2, Bv1, Bv2, Cv[16], tmps[16];
    memset(tmps, 0, sizeof(tmps));

    for (int k_idx = 0; k_idx < K; k_idx += bK9) {
        // 从A,B中取连续的4个数, 每个线程从A中取4个数, 从B中取4个数
        Av1 = *((float4 *)(&A[IDX9(row_a, col_a, lda)]));
        Bv1 = *((float4 *)(&B[IDX9(row_b, col_b, ldb)]));
        // 存到共享内存中, 每个线程存4个数, 存储位置连续
        ((float4 *)a_tile)[tx] = Av1;
        // ((float4 *)b_tile)[tx] = Bv;
        // 存在共享内存中, 每个线程存4个数, 但是存储位置不连续, 之间的距离为128
        b_tile[SIDX9(col_b, row_b)] = Bv1.x;
        b_tile[SIDX9(col_b, row_b+1)] = Bv1.y;
        b_tile[SIDX9(col_b, row_b+2)] = Bv1.z;
        b_tile[SIDX9(col_b, row_b+3)] = Bv1.w;
        A += lda8; // A偏移8列 8*lda (列优先)
        B += bK9;   // B偏移8行
        __syncthreads();
        
        #pragma unroll
        // 每个线程每次循环从共享内存A中读取8个数, 从共享内存B中读取8个数，然后进行8*8次累加计算; 共8bK9次循环
        for (int inner_k = 0; inner_k < bK9; ++inner_k) {   // bK9=8, 数据宽度为128
            Av1 = *((float4 *)(&a_tile[SIDX9(row_c, inner_k)])); // 从共享内存中取出连续的4个值给Av，需要和b相乘并累加
            Av2 = *((float4 *)(&a_tile[SIDX9(row_c+4, inner_k)]));
            Bv1 = *((float4 *)(&b_tile[SIDX9(col_c, inner_k)])); // 从共享内存中取出连续的4个值给Bv，这4个数对应bK8中每个维度的值
            Bv2 = *((float4 *)(&b_tile[SIDX9(col_c+4, inner_k)]));
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
    // row_c = range(0, 128, 8), col_c = range(0, 128, 8)
    Cv[0] = *((float4*)(&C[IDX9(row_c, col_c, ldc)]));
    Cv[1] = *((float4*)(&C[IDX9(row_c+4, col_c, ldc)]));
    Cv[2] = *((float4*)(&C[IDX9(row_c, col_c+1, ldc)]));
    Cv[3] = *((float4*)(&C[IDX9(row_c+4, col_c+1, ldc)]));
    Cv[4] = *((float4*)(&C[IDX9(row_c, col_c+2, ldc)]));
    Cv[5] = *((float4*)(&C[IDX9(row_c+4, col_c+2, ldc)]));
    Cv[6] = *((float4*)(&C[IDX9(row_c, col_c+3, ldc)]));
    Cv[7] = *((float4*)(&C[IDX9(row_c+4, col_c+3, ldc)]));
    Cv[8] = *((float4*)(&C[IDX9(row_c, col_c+4, ldc)]));
    Cv[9] = *((float4*)(&C[IDX9(row_c+4, col_c+4, ldc)]));
    Cv[10] = *((float4*)(&C[IDX9(row_c, col_c+5, ldc)]));
    Cv[11] = *((float4*)(&C[IDX9(row_c+4, col_c+5, ldc)]));
    Cv[12] = *((float4*)(&C[IDX9(row_c, col_c+6, ldc)]));
    Cv[13] = *((float4*)(&C[IDX9(row_c+4, col_c+6, ldc)]));
    Cv[14] = *((float4*)(&C[IDX9(row_c, col_c+7, ldc)]));
    Cv[15] = *((float4*)(&C[IDX9(row_c+4, col_c+7, ldc)]));
    
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
    // 将tmps中的值写回到C中
    *((float4*)(&C[IDX9(row_c, col_c, ldc)])) = tmps[0];
    *((float4*)(&C[IDX9(row_c+4, col_c, ldc)])) = tmps[1];
    *((float4*)(&C[IDX9(row_c, col_c+1, ldc)])) = tmps[2];
    *((float4*)(&C[IDX9(row_c+4, col_c+1, ldc)])) = tmps[3];
    *((float4*)(&C[IDX9(row_c, col_c+2, ldc)])) = tmps[4];
    *((float4*)(&C[IDX9(row_c+4, col_c+2, ldc)])) = tmps[5];
    *((float4*)(&C[IDX9(row_c, col_c+3, ldc)])) = tmps[6];
    *((float4*)(&C[IDX9(row_c+4, col_c+3, ldc)])) = tmps[7];
    *((float4*)(&C[IDX9(row_c, col_c+4, ldc)])) = tmps[8];
    *((float4*)(&C[IDX9(row_c+4, col_c+4, ldc)])) = tmps[9];
    *((float4*)(&C[IDX9(row_c, col_c+5, ldc)])) = tmps[10];
    *((float4*)(&C[IDX9(row_c+4, col_c+5, ldc)])) = tmps[11];
    *((float4*)(&C[IDX9(row_c, col_c+6, ldc)])) = tmps[12];
    *((float4*)(&C[IDX9(row_c+4, col_c+6, ldc)])) = tmps[13];
    *((float4*)(&C[IDX9(row_c, col_c+7, ldc)])) = tmps[14];
    *((float4*)(&C[IDX9(row_c+4, col_c+7, ldc)])) = tmps[15];
}
