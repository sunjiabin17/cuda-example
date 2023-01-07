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

#define bM12 128
#define bN12 128
#define bK12 8

#define IDX12(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX12(i,j) IDX12(i,j,(1<<7))     // shared memory index

// ({Ms,Ns,Ks}={128,128,8}, {Mr,Nr}={8,8}) {Mw,Nw}={4xMr,8xNr}
__global__ __launch_bounds__(256) void gemm12_8x8_double_buffer(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
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

    int row_c = (warp_row<<5) + (row_w<<3); // stride=8
    int col_c = (warp_col<<6) + (col_w<<3);

    int row_a = (tx&31)<<2, col_a = tx>>5;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    
    int K_upper = K >> 3;   // K_upper = K/8


    A = &(A[(bx<<7)]);
    B = &(B[(by<<7) * ldb]);
    C = &(C[(bx<<7) + (by<<7) * ldc]);


    __shared__ float a_tile[2][bM12*bK12];
    __shared__ float b_tile[2][bK12*bN12];
    float *ptr_a_tile, *ptr_b_tile;
    ptr_a_tile = (float*)a_tile;
    ptr_b_tile = (float*)b_tile;

    float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cv[16], tmps[16];
    float4 pref_Av, pref_Bv;
    float *ptr_A, *ptr_B;
    memset(tmps, 0, sizeof(tmps));

    // 从全局内存加载A和B中的一个block到共享内存
    pref_Av = *((float4 *)(&A[IDX12(row_a, col_a, lda)]));
    pref_Bv = *((float4 *)(&B[IDX12(row_b, col_b, ldb)]));
    ((float4 *)ptr_a_tile)[tx] = pref_Av;
    ptr_b_tile[SIDX12(col_b, row_b)] = pref_Bv.x;
    ptr_b_tile[SIDX12(col_b, row_b+1)] = pref_Bv.y;
    ptr_b_tile[SIDX12(col_b, row_b+2)] = pref_Bv.z;
    ptr_b_tile[SIDX12(col_b, row_b+3)] = pref_Bv.w;
    __syncthreads();
    Av1[0] = *((float4 *)(&ptr_a_tile[SIDX12(row_c, 0)]));
    Av2[0] = *((float4 *)(&ptr_a_tile[SIDX12(row_c+4, 0)]));
    Bv1[0] = *((float4 *)(&ptr_b_tile[SIDX12(col_c, 0)]));
    Bv2[0] = *((float4 *)(&ptr_b_tile[SIDX12(col_c+4, 0)]));

    for (int k_idx = 0; k_idx < K_upper; ++k_idx) {
        // packing A and B into shared memory
        int inc = (k_idx+1)%K_upper;
        int offset = ((k_idx+1)&1)<<10; // 切换到共享内存的第一行还是第二行
        ptr_A = A + inc*lda8;
        ptr_B = B + inc*bK12;
        // 预取全局内存中A和B中的下一个block
        pref_Av = *((float4 *)(&ptr_A[IDX12(row_a, col_a, lda)]));
        pref_Bv = *((float4 *)(&ptr_B[IDX12(row_b, col_b, ldb)]));
        #pragma unroll
        for (int inner_k = 0; inner_k < bK12; ++inner_k) {
            int next_inner_k = (inner_k+1)&7;
            // 预取共享内存中A和B的8个元素
            Av1[(inner_k+1)&1] = *((float4 *)(&ptr_a_tile[SIDX12(row_c, next_inner_k)]));
            Av2[(inner_k+1)&1] = *((float4 *)(&ptr_a_tile[SIDX12(row_c+4, next_inner_k)]));
            Bv1[(inner_k+1)&1] = *((float4 *)(&ptr_b_tile[SIDX12(col_c, next_inner_k)]));
            Bv2[(inner_k+1)&1] = *((float4 *)(&ptr_b_tile[SIDX12(col_c+4, next_inner_k)]));
            // 计算
            vscal(tmps[0], Av1[(inner_k)&1], Bv1[(inner_k)&1].x)
            vscal(tmps[1], Av2[(inner_k)&1], Bv1[(inner_k)&1].x)
            vscal(tmps[2], Av1[(inner_k)&1], Bv1[(inner_k)&1].y)
            vscal(tmps[3], Av2[(inner_k)&1], Bv1[(inner_k)&1].y)
            vscal(tmps[4], Av1[(inner_k)&1], Bv1[(inner_k)&1].z)
            vscal(tmps[5], Av2[(inner_k)&1], Bv1[(inner_k)&1].z)
            vscal(tmps[6], Av1[(inner_k)&1], Bv1[(inner_k)&1].w)
            vscal(tmps[7], Av2[(inner_k)&1], Bv1[(inner_k)&1].w)
            vscal(tmps[8], Av1[(inner_k)&1], Bv2[(inner_k)&1].x)
            vscal(tmps[9], Av2[(inner_k)&1], Bv2[(inner_k)&1].x)
            vscal(tmps[10], Av1[(inner_k)&1], Bv2[(inner_k)&1].y)
            vscal(tmps[11], Av2[(inner_k)&1], Bv2[(inner_k)&1].y)
            vscal(tmps[12], Av1[(inner_k)&1], Bv2[(inner_k)&1].z)
            vscal(tmps[13], Av2[(inner_k)&1], Bv2[(inner_k)&1].z)
            vscal(tmps[14], Av1[(inner_k)&1], Bv2[(inner_k)&1].w)
            vscal(tmps[15], Av2[(inner_k)&1], Bv2[(inner_k)&1].w)
        }
        // 这里把数据存到共享内存的另一块buffer中
        // 减少一次同步 __syncthreads()
        ptr_a_tile = (float *)a_tile + offset;
        ptr_b_tile = (float *)b_tile + offset;
        ((float4 *)ptr_a_tile)[tx] = pref_Av;
        ptr_b_tile[SIDX12(col_b, row_b)] = pref_Bv.x;
        ptr_b_tile[SIDX12(col_b, row_b+1)] = pref_Bv.y;
        ptr_b_tile[SIDX12(col_b, row_b+2)] = pref_Bv.z;
        ptr_b_tile[SIDX12(col_b, row_b+3)] = pref_Bv.w;
        __syncthreads();

        // 下一次循环需要的共享内存中的8个元素
        Av1[0] = *((float4 *)(&ptr_a_tile[SIDX12(row_c, 0)]));
        Av2[0] = *((float4 *)(&ptr_a_tile[SIDX12(row_c+4, 0)]));
        Bv1[0] = *((float4 *)(&ptr_b_tile[SIDX12(col_c, 0)]));
        Bv2[0] = *((float4 *)(&ptr_b_tile[SIDX12(col_c+4, 0)]));
    }
    Cv[0] = *((float4 *)(&C[IDX12(row_c, col_c, ldc)]));
    Cv[1] = *((float4 *)(&C[IDX12(row_c+4, col_c, ldc)]));
    Cv[2] = *((float4 *)(&C[IDX12(row_c, col_c+1, ldc)]));
    Cv[3] = *((float4 *)(&C[IDX12(row_c+4, col_c+1, ldc)]));
    Cv[4] = *((float4 *)(&C[IDX12(row_c, col_c+2, ldc)]));
    Cv[5] = *((float4 *)(&C[IDX12(row_c+4, col_c+2, ldc)]));
    Cv[6] = *((float4 *)(&C[IDX12(row_c, col_c+3, ldc)]));
    Cv[7] = *((float4 *)(&C[IDX12(row_c+4, col_c+3, ldc)]));
    Cv[8] = *((float4 *)(&C[IDX12(row_c, col_c+4, ldc)]));
    Cv[9] = *((float4 *)(&C[IDX12(row_c+4, col_c+4, ldc)]));
    Cv[10] = *((float4 *)(&C[IDX12(row_c, col_c+5, ldc)]));
    Cv[11] = *((float4 *)(&C[IDX12(row_c+4, col_c+5, ldc)]));
    Cv[12] = *((float4 *)(&C[IDX12(row_c, col_c+6, ldc)]));
    Cv[13] = *((float4 *)(&C[IDX12(row_c+4, col_c+6, ldc)]));
    Cv[14] = *((float4 *)(&C[IDX12(row_c, col_c+7, ldc)]));
    Cv[15] = *((float4 *)(&C[IDX12(row_c+4, col_c+7, ldc)]));

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

    *((float4 *)(&C[IDX12(row_c, col_c, ldc)])) = tmps[0];
    *((float4 *)(&C[IDX12(row_c+4, col_c, ldc)])) = tmps[1];
    *((float4 *)(&C[IDX12(row_c, col_c+1, ldc)])) = tmps[2];
    *((float4 *)(&C[IDX12(row_c+4, col_c+1, ldc)])) = tmps[3];
    *((float4 *)(&C[IDX12(row_c, col_c+2, ldc)])) = tmps[4];
    *((float4 *)(&C[IDX12(row_c+4, col_c+2, ldc)])) = tmps[5];
    *((float4 *)(&C[IDX12(row_c, col_c+3, ldc)])) = tmps[6];
    *((float4 *)(&C[IDX12(row_c+4, col_c+3, ldc)])) = tmps[7];
    *((float4 *)(&C[IDX12(row_c, col_c+4, ldc)])) = tmps[8];
    *((float4 *)(&C[IDX12(row_c+4, col_c+4, ldc)])) = tmps[9];
    *((float4 *)(&C[IDX12(row_c, col_c+5, ldc)])) = tmps[10];
    *((float4 *)(&C[IDX12(row_c+4, col_c+5, ldc)])) = tmps[11];
    *((float4 *)(&C[IDX12(row_c, col_c+6, ldc)])) = tmps[12];
    *((float4 *)(&C[IDX12(row_c+4, col_c+6, ldc)])) = tmps[13];
    *((float4 *)(&C[IDX12(row_c, col_c+7, ldc)])) = tmps[14];
    *((float4 *)(&C[IDX12(row_c+4, col_c+7, ldc)])) = tmps[15];

}