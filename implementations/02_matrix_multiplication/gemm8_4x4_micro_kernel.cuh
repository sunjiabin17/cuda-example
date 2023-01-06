#define bM8 64
#define bN8 64
#define bK8 16
#define IDX8(i,j,ld) (((j)*(ld))+(i))  // A,B,C matrix index
#define SIDX8(i,j) IDX8(i,j,(1<<6))     // shared memory index

// ({Ms,Ns,Ks}={64,64,16}, {Mr,Nr}={4,4})
__global__ __launch_bounds__(256) void gemm8_4x4_micro_kernel(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    // block size: 64*64, thread size: 16*16
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x; // one dim
    int bx = blockIdx.x, by = blockIdx.y;
    

    int row_a = (tx&15)<<2, col_a = tx>>4;  // row_a = range(0, 64, 4), col_a = range(0, 16, 1)
    int row_b = (tx&3)<<2, col_b = tx>>2;   // row_b = range(0, 16, 4), col_b = range(0, 64, 1)
    int col_c = col_a<<2;                   // col_c = range(0, 64, 4)
    int lda16 = lda<<4;

    A = &(A[(bx << 6)]);
    B = &(B[(by << 6) * ldb]);
    C = &(C[(bx << 6) + (by << 6) * ldc]);  // the TB size is 64.

    __shared__ float a_tile[bM8*bK8];
    __shared__ float b_tile[bK8*bN8];

    float4 Av, Bv, Cv[4], tmps[4];
    memset(tmps, 0, sizeof(tmps));

    // 1. 数据布局如何变化: from global memory to shared memory
    // 2. float4加载连续内存如何处理
    // 3. 数据在shared memory中的位置以及如何计算
    for (int k_idx = 0; k_idx < K; k_idx += bK8) {
        Av = *((float4 *)(&A[IDX8(row_a, col_a, lda)]));   // 从A中取连续的4个数
        Bv = *((float4 *)(&B[IDX8(row_b, col_b, ldb)]));   // 从B中取连续的4个数
        ((float4 *)a_tile)[tx] = Av;            // 将Av中的4个数存入共享内存中的连续内存
        // ((float4 *)b_tile)[tx] = Bv;
        b_tile[SIDX8(col_b, row_b)] = Bv.x;     // 将Bv中的4个数存入共享内存, 内存不连续, 之间的距离为64
        b_tile[SIDX8(col_b, row_b+1)] = Bv.y;
        b_tile[SIDX8(col_b, row_b+2)] = Bv.z;
        b_tile[SIDX8(col_b, row_b+3)] = Bv.w;
        A += lda16; // A偏移16列
        B += bK8;   // B偏移16行
        __syncthreads();
        #pragma unroll
        for (int inner_k = 0; inner_k < bK8; ++inner_k) {   // bK8=16, 数据宽度为64, 每次计算4*4个数
            Av = *((float4 *)(&a_tile[SIDX8(row_a, inner_k)])); // 从共享内存中取出连续的4个值给Av，需要和b相乘并累加
            Bv = *((float4 *)(&b_tile[SIDX8(col_c, inner_k)])); // 从共享内存中取出连续的4个值给Bv，这4个数对应bK8中每个维度的值
            // 第一次迭代: inner_k=0, 
            // 当线程是0-3时, 
            // Av从shared memory中取的地址为0,1,2,3;4,...,15; (第一列0:32)  (列优先)
            // Bv从shared memory中取的地址为0,1,2,3.0,1,2,3,...,0,1,2,3
            // 当线程是4-7时,
            // Av从shared memory中取的地址为16,17,18,19;20,...,31;(第一列16:32)
            // Bv从shared memory中取的地址为0,1,2,3.0,1,2,3,...,0,1,2,3
            // ...
            // 第二次迭代: inner_k=1, 
            // 当线程是0-3时,
            // Av从shared memory中取的地址为64,65,66,67;68,...,79; (第二列0:32)
            // Bv从shared memory中取的地址为4,5,6,7.4,5,6,7,...,4,5,6,7
            // 当线程是4-7时,
            // Av从shared memory中取的地址为80,81,82,83;84,...,95;(第二列16:32)
            // Bv从shared memory中取的地址为4,5,6,7.4,5,6,7,...,4,5,6,7
            // ...
            tmps[0].x += Av.x * Bv.x;   // tmps[x][y] 对应计算A[x][:] * B[:][y], 结果存在C[x][y]处
            tmps[0].y += Av.y * Bv.x;
            tmps[0].z += Av.z * Bv.x;
            tmps[0].w += Av.w * Bv.x;
            tmps[1].x += Av.x * Bv.y;
            tmps[1].y += Av.y * Bv.y;
            tmps[1].z += Av.z * Bv.y;
            tmps[1].w += Av.w * Bv.y;
            tmps[2].x += Av.x * Bv.z;
            tmps[2].y += Av.y * Bv.z;
            tmps[2].z += Av.z * Bv.z;
            tmps[2].w += Av.w * Bv.z;
            tmps[3].x += Av.x * Bv.w;
            tmps[3].y += Av.y * Bv.w;
            tmps[3].z += Av.z * Bv.w;
            tmps[3].w += Av.w * Bv.w;
        }
        __syncthreads();
    }
    // 从全局内存中取出row_a, col_c处连续的4个值给Cv
    // row_a = range(0, 64, 4), col_c = range(0, 64, 4)
    Cv[0] = *((float4*)(&C[IDX8(row_a, col_c, ldc)]));      // 对应(row_a, col_c), (row_a+1, col_c), (row_a+2, col_c), (row_a+3, col_c)
    Cv[1] = *((float4*)(&C[IDX8(row_a, col_c+1, ldc)]));    
    Cv[2] = *((float4*)(&C[IDX8(row_a, col_c+2, ldc)]));
    Cv[3] = *((float4*)(&C[IDX8(row_a, col_c+3, ldc)]));

    tmps[0].x = alpha * tmps[0].x + beta * Cv[0].x;
    tmps[0].y = alpha * tmps[0].y + beta * Cv[0].y;
    tmps[0].z = alpha * tmps[0].z + beta * Cv[0].z;
    tmps[0].w = alpha * tmps[0].w + beta * Cv[0].w;
    tmps[1].x = alpha * tmps[1].x + beta * Cv[1].x;
    tmps[1].y = alpha * tmps[1].y + beta * Cv[1].y;
    tmps[1].z = alpha * tmps[1].z + beta * Cv[1].z;
    tmps[1].w = alpha * tmps[1].w + beta * Cv[1].w;
    tmps[2].x = alpha * tmps[2].x + beta * Cv[2].x;
    tmps[2].y = alpha * tmps[2].y + beta * Cv[2].y;
    tmps[2].z = alpha * tmps[2].z + beta * Cv[2].z;
    tmps[2].w = alpha * tmps[2].w + beta * Cv[2].w;
    tmps[3].x = alpha * tmps[3].x + beta * Cv[3].x;
    tmps[3].y = alpha * tmps[3].y + beta * Cv[3].y;
    tmps[3].z = alpha * tmps[3].z + beta * Cv[3].z;
    tmps[3].w = alpha * tmps[3].w + beta * Cv[3].w;

    // tmps[x][1,2,3,4]对应C[x][1,2,3,4]
    *((float4*)(&C[IDX8(row_a, col_c, ldc)])) = tmps[0];
    *((float4*)(&C[IDX8(row_a, col_c+1, ldc)])) = tmps[1];
    *((float4*)(&C[IDX8(row_a, col_c+2, ldc)])) = tmps[2];
    *((float4*)(&C[IDX8(row_a, col_c+3, ldc)])) = tmps[3];
}

