#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
// naive version
__global__  __launch_bounds__(1024)
void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count++){
        tmp += A(tx, k_count) * B(k_count, ty);
    }
    C(tx,ty) = alpha * tmp + beta*C(tx,ty);
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa(i,j) sa[((i)<<5) + (j)]
#define sb(i,j) sb[((i)<<5) + (j)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
__global__  __launch_bounds__(1024)
void mysgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa[MS*KS];
    __shared__ float sb[KS*NS];
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa(tx,ty)=A(tx,ty);
        sb(ty,tx)=B(tx,ty);
        A+=(lda<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            tmp += sa(tx,inner_k_count) * sb(ty,inner_k_count);
        }
        __syncthreads();
    }
    C(tx,ty) = alpha * tmp + beta*C(tx,ty);
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa(i,j) sa[((i)<<5) + (j)]
#define sb(i,j) sb[((i)<<5) + (j)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// save one living register ty.
__global__  __launch_bounds__(1024)
void mysgemm_v3(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa[MS*KS];
    __shared__ float sb[KS*NS];
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa(row,col)=A(row,col);
        sb(col,row)=B(row,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            tmp += sa(row,inner_k_count) * sb(col,inner_k_count);
        }
        __syncthreads();
    }
    C(row,col) = alpha * tmp + beta*C(row,col);
}


#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa4(i,j) sa4[((j)<<5) + (i)]
#define sb4(i,j) sb4[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
__global__  __launch_bounds__(1024)
void mysgemm_v4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa4[MS*KS];
    __shared__ float sb4[KS*NS];
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa4(row,col)=A(row,col);
        sb4(col,row)=B(row,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            tmp += sa4(row,inner_k_count) * sb4(col,inner_k_count);
        }
        __syncthreads();
    }
    C(row,col) = alpha * tmp + beta*C(row,col);
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa5(i,j) sa5[((j)<<5) + (i)]
#define sb5(i,j) sb5[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
__global__  __launch_bounds__(256)
void mysgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3, col = tx>>3;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa5[MS*KS];
    __shared__ float sb5[KS*NS];
    float Cres[4] = {0., 0., 0., 0.};
    float b00;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa5(row1,col)=A(row1,col);
        sa5(row2,col)=A(row2,col);
        sa5(row3,col)=A(row3,col);
        sa5(row4,col)=A(row4,col);
        sb5(col,row1)=B(row1,col);
        sb5(col,row2)=B(row2,col);
        sb5(col,row3)=B(row3,col);
        sb5(col,row4)=B(row4,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            b00 = sb5(col,inner_k_count);
            Cres[0] += sa5(row1,inner_k_count) * b00;
            Cres[1] += sa5(row2,inner_k_count) * b00;
            Cres[2] += sa5(row3,inner_k_count) * b00;
            Cres[3] += sa5(row4,inner_k_count) * b00;
        }
        __syncthreads();
    }
    C(row1,col) = alpha * Cres[0] + beta*C(row1,col);
    C(row2,col) = alpha * Cres[1] + beta*C(row2,col);
    C(row3,col) = alpha * Cres[2] + beta*C(row3,col);
    C(row4,col) = alpha * Cres[3] + beta*C(row4,col);
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa7(i,j) sa7[((j)<<6) + (i)]
#define sb7(i,j) sb7[((j)<<6) + (i)]
#define MS_7 64
#define NS_7 64
#define KS_7 16
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x4 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v7(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row_a = (tx&15)<<2, col_a = tx>>4;
    int row_b = (tx&3)<<2, col_b = tx>>2;
    int col_c = col_a<<2;
    int lda16 = lda<<4;
    A = &A((bx<<6),0);
    B = &B(0,(by<<6));
    C = &C((bx<<6),(by<<6));//the TB size is 64.
    __shared__ float sa7[1024];
    __shared__ float sb7[1024];
    float4 Av, Bv, Cv[4], Cres[4];
    memset(Cres, 0, sizeof(Cres));
    for (int k_count = 0; k_count<K; k_count+=KS_7){
        vload(Av, &A(row_a,col_a))
        vload(Bv, &B(row_b,col_b))
        ((float4 *)sa7)[tx] = Av;
        sb7(col_b,row_b)=Bv.x;
        sb7(col_b,row_b+1)=Bv.y;
        sb7(col_b,row_b+2)=Bv.z;
        sb7(col_b,row_b+3)=Bv.w;
        A+=lda16;B+=16;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_7;inner_k_count++){
            vload(Av, &sa7(row_a,inner_k_count))
            vload(Bv, &sb7(col_c,inner_k_count))
            vscal(Cres[0], Av, Bv.x)
            vscal(Cres[1], Av, Bv.y)
            vscal(Cres[2], Av, Bv.z)
            vscal(Cres[3], Av, Bv.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_a,col_c))
    vload(Cv[1], &C(row_a,col_c+1))
    vload(Cv[2], &C(row_a,col_c+2))
    vload(Cv[3], &C(row_a,col_c+3))
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    vstore(&C(row_a,col_c), Cres[0])
    vstore(&C(row_a,col_c+1), Cres[1])
    vstore(&C(row_a,col_c+2), Cres[2])
    vstore(&C(row_a,col_c+3), Cres[3])
}


#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa8(i,j) sa8[((j)<<7) + (i)]
#define sb8(i,j) sb8[((j)<<7) + (i)]
#define MS_8 128
#define NS_8 128
#define KS_8 8
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v8(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row_a = (tx&31)<<2, col_a = tx>>5;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    int row_c = (tx&15)<<3, col_c = (tx>>4)<<3;
    A = &A((bx<<7),0);
    B = &B(0,(by<<7));
    C = &C((bx<<7),(by<<7));//the TB size is 128.
    __shared__ float sa8[1024];
    __shared__ float sb8[1024];
    float4 Av1, Av2, Bv1, Bv2, Cv[16], Cres[16];
    memset(Cres, 0, sizeof(Cres));//clear registers
    for (int k_count = 0; k_count<K; k_count+=KS_8){
        vload(Av1, &A(row_a,col_a))
        vload(Bv1, &B(row_b,col_b))
        ((float4 *)sa8)[tx] = Av1;
        sb8(col_b,row_b)=Bv1.x;
        sb8(col_b,row_b+1)=Bv1.y;
        sb8(col_b,row_b+2)=Bv1.z;
        sb8(col_b,row_b+3)=Bv1.w;
        A+=lda8;B+=8;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_8;inner_k_count++){
            vload(Av1, &sa8(row_c,inner_k_count))
            vload(Av2, &sa8(row_c+4,inner_k_count))
            vload(Bv1, &sb8(col_c,inner_k_count))
            vload(Bv2, &sb8(col_c+4,inner_k_count))
            vscal(Cres[0], Av1, Bv1.x)
            vscal(Cres[1], Av2, Bv1.x)
            vscal(Cres[2], Av1, Bv1.y)
            vscal(Cres[3], Av2, Bv1.y)
            vscal(Cres[4], Av1, Bv1.z)
            vscal(Cres[5], Av2, Bv1.z)
            vscal(Cres[6], Av1, Bv1.w)
            vscal(Cres[7], Av2, Bv1.w)
            vscal(Cres[8], Av1, Bv2.x)
            vscal(Cres[9], Av2, Bv2.x)
            vscal(Cres[10], Av1, Bv2.y)
            vscal(Cres[11], Av2, Bv2.y)
            vscal(Cres[12], Av1, Bv2.z)
            vscal(Cres[13], Av2, Bv2.z)
            vscal(Cres[14], Av1, Bv2.w)
            vscal(Cres[15], Av2, Bv2.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_c,col_c))
    vload(Cv[1], &C(row_c+4,col_c))
    vload(Cv[2], &C(row_c,col_c+1))
    vload(Cv[3], &C(row_c+4,col_c+1))
    vload(Cv[4], &C(row_c,col_c+2))
    vload(Cv[5], &C(row_c+4,col_c+2))
    vload(Cv[6], &C(row_c,col_c+3))
    vload(Cv[7], &C(row_c+4,col_c+3))
    vload(Cv[8], &C(row_c,col_c+4))
    vload(Cv[9], &C(row_c+4,col_c+4))
    vload(Cv[10], &C(row_c,col_c+5))
    vload(Cv[11], &C(row_c+4,col_c+5))
    vload(Cv[12], &C(row_c,col_c+6))
    vload(Cv[13], &C(row_c+4,col_c+6))
    vload(Cv[14], &C(row_c,col_c+7))
    vload(Cv[15], &C(row_c+4,col_c+7))
    
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
    simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
    simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
    simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

    simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
    simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
    simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
    simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

    simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
    simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
    simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
    simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

    vstore(&C(row_c,col_c), Cres[0])
    vstore(&C(row_c+4,col_c), Cres[1])
    vstore(&C(row_c,col_c+1), Cres[2])
    vstore(&C(row_c+4,col_c+1), Cres[3])
    vstore(&C(row_c,col_c+2), Cres[4])
    vstore(&C(row_c+4,col_c+2), Cres[5])
    vstore(&C(row_c,col_c+3), Cres[6])
    vstore(&C(row_c+4,col_c+3), Cres[7])
    vstore(&C(row_c,col_c+4), Cres[8])
    vstore(&C(row_c+4,col_c+4), Cres[9])
    vstore(&C(row_c,col_c+5), Cres[10])
    vstore(&C(row_c+4,col_c+5), Cres[11])
    vstore(&C(row_c,col_c+6), Cres[12])
    vstore(&C(row_c+4,col_c+6), Cres[13])
    vstore(&C(row_c,col_c+7), Cres[14])
    vstore(&C(row_c+4,col_c+7), Cres[15])
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa9(i,j) sa9[((j)<<7) + (i)]
#define sb9(i,j) sb9[((j)<<7) + (i)]
#define MS_9 128
#define NS_9 128
#define KS_9 8
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v9(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    int row_c = (warp_row<<5) + (row_w<<3), col_c = (warp_col<<6) + (col_w<<3);
    int row_a = (tx&31)<<2, col_a = tx>>5;
    A = &A((bx<<7),0);
    B = &B(0,(by<<7));
    C = &C((bx<<7),(by<<7));//the TB size is 128.
    __shared__ float sa9[1024];
    __shared__ float sb9[1024];
    float4 Av1, Av2, Bv1, Bv2, Cv[16], Cres[16];
    memset(Cres, 0, sizeof(Cres));//clear registers
    for (int k_count = 0; k_count<K; k_count+=KS_9){
        /*packing A and B into shared memory*/
        vload(Av1, &A(row_a,col_a))
        vload(Bv1, &B(row_b,col_b))
        ((float4 *)sa9)[tx] = Av1;
        sb9(col_b,row_b)=Bv1.x;
        sb9(col_b,row_b+1)=Bv1.y;
        sb9(col_b,row_b+2)=Bv1.z;
        sb9(col_b,row_b+3)=Bv1.w;
        A+=lda8;B+=8;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_9;inner_k_count++){
            vload(Av1, &sa9(row_c,inner_k_count))
            vload(Av2, &sa9(row_c+4,inner_k_count))
            vload(Bv1, &sb9(col_c,inner_k_count))
            vload(Bv2, &sb9(col_c+4,inner_k_count))
            vscal(Cres[0], Av1, Bv1.x)
            vscal(Cres[1], Av2, Bv1.x)
            vscal(Cres[2], Av1, Bv1.y)
            vscal(Cres[3], Av2, Bv1.y)
            vscal(Cres[4], Av1, Bv1.z)
            vscal(Cres[5], Av2, Bv1.z)
            vscal(Cres[6], Av1, Bv1.w)
            vscal(Cres[7], Av2, Bv1.w)
            vscal(Cres[8], Av1, Bv2.x)
            vscal(Cres[9], Av2, Bv2.x)
            vscal(Cres[10], Av1, Bv2.y)
            vscal(Cres[11], Av2, Bv2.y)
            vscal(Cres[12], Av1, Bv2.z)
            vscal(Cres[13], Av2, Bv2.z)
            vscal(Cres[14], Av1, Bv2.w)
            vscal(Cres[15], Av2, Bv2.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_c,col_c))
    vload(Cv[1], &C(row_c+4,col_c))
    vload(Cv[2], &C(row_c,col_c+1))
    vload(Cv[3], &C(row_c+4,col_c+1))
    vload(Cv[4], &C(row_c,col_c+2))
    vload(Cv[5], &C(row_c+4,col_c+2))
    vload(Cv[6], &C(row_c,col_c+3))
    vload(Cv[7], &C(row_c+4,col_c+3))
    vload(Cv[8], &C(row_c,col_c+4))
    vload(Cv[9], &C(row_c+4,col_c+4))
    vload(Cv[10], &C(row_c,col_c+5))
    vload(Cv[11], &C(row_c+4,col_c+5))
    vload(Cv[12], &C(row_c,col_c+6))
    vload(Cv[13], &C(row_c+4,col_c+6))
    vload(Cv[14], &C(row_c,col_c+7))
    vload(Cv[15], &C(row_c+4,col_c+7))
    
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
    simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
    simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
    simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

    simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
    simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
    simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
    simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

    simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
    simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
    simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
    simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

    vstore(&C(row_c,col_c), Cres[0])
    vstore(&C(row_c+4,col_c), Cres[1])
    vstore(&C(row_c,col_c+1), Cres[2])
    vstore(&C(row_c+4,col_c+1), Cres[3])
    vstore(&C(row_c,col_c+2), Cres[4])
    vstore(&C(row_c+4,col_c+2), Cres[5])
    vstore(&C(row_c,col_c+3), Cres[6])
    vstore(&C(row_c+4,col_c+3), Cres[7])
    vstore(&C(row_c,col_c+4), Cres[8])
    vstore(&C(row_c+4,col_c+4), Cres[9])
    vstore(&C(row_c,col_c+5), Cres[10])
    vstore(&C(row_c+4,col_c+5), Cres[11])
    vstore(&C(row_c,col_c+6), Cres[12])
    vstore(&C(row_c+4,col_c+6), Cres[13])
    vstore(&C(row_c,col_c+7), Cres[14])
    vstore(&C(row_c+4,col_c+7), Cres[15])
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define ptr_A(i,j) ptr_A[(i) + (j)*lda]
#define ptr_B(i,j) ptr_B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa10(i,j) sa10[((j)<<7) + (i)]
#define sb10(i,j) sb10[((j)<<7) + (i)]
#define MS_10 128
#define NS_10 128
#define KS_10 8
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v10(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    int row_c = (warp_row<<5) + (row_w<<3), col_c = (warp_col<<6) + (col_w<<3);
    int row_a = (tx&31)<<2, col_a = tx>>5;
    int K_upper = K>>3;
    A = &A((bx<<7),0);
    B = &B(0,(by<<7));
    C = &C((bx<<7),(by<<7));//the TB size is 128.
    __shared__ float sa10[1024];
    __shared__ float sb10[1024];
    float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cv[16], Cres[16];
    float4 pref_Av, pref_Bv;
    float* ptr_A, *ptr_B;
    memset(Cres, 0, sizeof(Cres));//clear registers
    vload(pref_Av, &A(row_a,col_a))
    vload(pref_Bv, &B(row_b,col_b))
    ((float4 *)sa10)[tx] = pref_Av;
    sb10(col_b,row_b)=pref_Bv.x;
    sb10(col_b,row_b+1)=pref_Bv.y;
    sb10(col_b,row_b+2)=pref_Bv.z;
    sb10(col_b,row_b+3)=pref_Bv.w;
    __syncthreads();
    vload(Av1[0], &sa10(row_c,0))
    vload(Av2[0], &sa10(row_c+4,0))
    vload(Bv1[0], &sb10(col_c,0))
    vload(Bv2[0], &sb10(col_c+4,0))
    for (int k_count = 0; k_count<K_upper; k_count++){
        /*packing A and B into shared memory*/
        int inc = (k_count+1)%K_upper;
        ptr_A = A + inc * lda8;
        ptr_B = B + inc * 8;
        vload(pref_Av, &ptr_A(row_a,col_a))
        vload(pref_Bv, &ptr_B(row_b,col_b))
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_10;inner_k_count++){
            int next_inner_k_count = (inner_k_count+1)&7;
            vload(Av1[(inner_k_count+1)&1], &sa10(row_c,next_inner_k_count))
            vload(Av2[(inner_k_count+1)&1], &sa10(row_c+4,next_inner_k_count))
            vload(Bv1[(inner_k_count+1)&1], &sb10(col_c,next_inner_k_count))
            vload(Bv2[(inner_k_count+1)&1], &sb10(col_c+4,next_inner_k_count))
            vscal(Cres[0], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].x)
            vscal(Cres[1], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].x)
            vscal(Cres[2], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].y)
            vscal(Cres[3], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].y)
            vscal(Cres[4], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].z)
            vscal(Cres[5], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].z)
            vscal(Cres[6], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].w)
            vscal(Cres[7], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].w)
            vscal(Cres[8], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].x)
            vscal(Cres[9], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].x)
            vscal(Cres[10], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].y)
            vscal(Cres[11], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].y)
            vscal(Cres[12], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].z)
            vscal(Cres[13], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].z)
            vscal(Cres[14], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].w)
            vscal(Cres[15], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].w)
        }
        __syncthreads();
        ((float4 *)sa10)[tx] = pref_Av;
        sb10(col_b,row_b)=pref_Bv.x;
        sb10(col_b,row_b+1)=pref_Bv.y;
        sb10(col_b,row_b+2)=pref_Bv.z;
        sb10(col_b,row_b+3)=pref_Bv.w;
        __syncthreads();
        vload(Av1[0], &sa10(row_c,0))
        vload(Av2[0], &sa10(row_c+4,0))
        vload(Bv1[0], &sb10(col_c,0))
        vload(Bv2[0], &sb10(col_c+4,0))
    }
    vload(Cv[0], &C(row_c,col_c))
    vload(Cv[1], &C(row_c+4,col_c))
    vload(Cv[2], &C(row_c,col_c+1))
    vload(Cv[3], &C(row_c+4,col_c+1))
    vload(Cv[4], &C(row_c,col_c+2))
    vload(Cv[5], &C(row_c+4,col_c+2))
    vload(Cv[6], &C(row_c,col_c+3))
    vload(Cv[7], &C(row_c+4,col_c+3))
    vload(Cv[8], &C(row_c,col_c+4))
    vload(Cv[9], &C(row_c+4,col_c+4))
    vload(Cv[10], &C(row_c,col_c+5))
    vload(Cv[11], &C(row_c+4,col_c+5))
    vload(Cv[12], &C(row_c,col_c+6))
    vload(Cv[13], &C(row_c+4,col_c+6))
    vload(Cv[14], &C(row_c,col_c+7))
    vload(Cv[15], &C(row_c+4,col_c+7))
    
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
    simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
    simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
    simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

    simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
    simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
    simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
    simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

    simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
    simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
    simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
    simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

    vstore(&C(row_c,col_c), Cres[0])
    vstore(&C(row_c+4,col_c), Cres[1])
    vstore(&C(row_c,col_c+1), Cres[2])
    vstore(&C(row_c+4,col_c+1), Cres[3])
    vstore(&C(row_c,col_c+2), Cres[4])
    vstore(&C(row_c+4,col_c+2), Cres[5])
    vstore(&C(row_c,col_c+3), Cres[6])
    vstore(&C(row_c+4,col_c+3), Cres[7])
    vstore(&C(row_c,col_c+4), Cres[8])
    vstore(&C(row_c+4,col_c+4), Cres[9])
    vstore(&C(row_c,col_c+5), Cres[10])
    vstore(&C(row_c+4,col_c+5), Cres[11])
    vstore(&C(row_c,col_c+6), Cres[12])
    vstore(&C(row_c+4,col_c+6), Cres[13])
    vstore(&C(row_c,col_c+7), Cres[14])
    vstore(&C(row_c+4,col_c+7), Cres[15])
}


#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define ptr_A(i,j) ptr_A[(i) + (j)*lda]
#define ptr_B(i,j) ptr_B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa11(i,j) sa11[((j)<<7) + (i)]
#define sb11(i,j) sb11[((j)<<7) + (i)]
#define ptr_sa11(i,j) ptr_sa11[((j)<<7) + (i)]
#define ptr_sb11(i,j) ptr_sb11[((j)<<7) + (i)]
#define MS_11 128
#define NS_11 128
#define KS_11 8
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v11(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    int row_c = (warp_row<<5) + (row_w<<3), col_c = (warp_col<<6) + (col_w<<3);
    int row_a = (tx&31)<<2, col_a = tx>>5;
    int K_upper = K>>3;
    A = &A((bx<<7),0);
    B = &B(0,(by<<7));
    C = &C((bx<<7),(by<<7));//the TB size is 128.
    __shared__ float sa11[2][1024];
    __shared__ float sb11[2][1024];
    float *ptr_sa11, *ptr_sb11;
    ptr_sa11 = (float*)sa11;
    ptr_sb11 = (float*)sb11;
    float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cv[16], Cres[16];
    float4 pref_Av, pref_Bv;
    float* ptr_A, *ptr_B;
    memset(Cres, 0, sizeof(Cres));//clear registers
    vload(pref_Av, &A(row_a,col_a))
    vload(pref_Bv, &B(row_b,col_b))
    ((float4 *)ptr_sa11)[tx] = pref_Av;
    ptr_sb11(col_b,row_b)=pref_Bv.x;
    ptr_sb11(col_b,row_b+1)=pref_Bv.y;
    ptr_sb11(col_b,row_b+2)=pref_Bv.z;
    ptr_sb11(col_b,row_b+3)=pref_Bv.w;
    __syncthreads();
    vload(Av1[0], &ptr_sa11(row_c,0))
    vload(Av2[0], &ptr_sa11(row_c+4,0))
    vload(Bv1[0], &ptr_sb11(col_c,0))
    vload(Bv2[0], &ptr_sb11(col_c+4,0))
    for (int k_count = 0; k_count<K_upper; k_count++){
        /*packing A and B into shared memory*/
        int inc = (k_count+1)%K_upper;
        int offset = ((k_count+1)&1)<<10;
        ptr_A = A + inc * lda8;
        ptr_B = B + inc * 8;
        vload(pref_Av, &ptr_A(row_a,col_a))
        vload(pref_Bv, &ptr_B(row_b,col_b))
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_11;inner_k_count++){
            int next_inner_k_count = (inner_k_count+1)&7;
            vload(Av1[(inner_k_count+1)&1], &ptr_sa11(row_c,next_inner_k_count))
            vload(Av2[(inner_k_count+1)&1], &ptr_sa11(row_c+4,next_inner_k_count))
            vload(Bv1[(inner_k_count+1)&1], &ptr_sb11(col_c,next_inner_k_count))
            vload(Bv2[(inner_k_count+1)&1], &ptr_sb11(col_c+4,next_inner_k_count))
            vscal(Cres[0], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].x)
            vscal(Cres[1], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].x)
            vscal(Cres[2], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].y)
            vscal(Cres[3], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].y)
            vscal(Cres[4], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].z)
            vscal(Cres[5], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].z)
            vscal(Cres[6], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].w)
            vscal(Cres[7], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].w)
            vscal(Cres[8], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].x)
            vscal(Cres[9], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].x)
            vscal(Cres[10], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].y)
            vscal(Cres[11], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].y)
            vscal(Cres[12], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].z)
            vscal(Cres[13], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].z)
            vscal(Cres[14], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].w)
            vscal(Cres[15], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].w)
        }
        ptr_sa11 = (float*)sa11 + offset;
        ptr_sb11 = (float*)sb11 + offset;
        ((float4 *)ptr_sa11)[tx] = pref_Av;
        ptr_sb11(col_b,row_b)=pref_Bv.x;
        ptr_sb11(col_b,row_b+1)=pref_Bv.y;
        ptr_sb11(col_b,row_b+2)=pref_Bv.z;
        ptr_sb11(col_b,row_b+3)=pref_Bv.w;
        __syncthreads();
        vload(Av1[0], &ptr_sa11(row_c,0))
        vload(Av2[0], &ptr_sa11(row_c+4,0))
        vload(Bv1[0], &ptr_sb11(col_c,0))
        vload(Bv2[0], &ptr_sb11(col_c+4,0))
    }
    vload(Cv[0], &C(row_c,col_c))
    vload(Cv[1], &C(row_c+4,col_c))
    vload(Cv[2], &C(row_c,col_c+1))
    vload(Cv[3], &C(row_c+4,col_c+1))
    vload(Cv[4], &C(row_c,col_c+2))
    vload(Cv[5], &C(row_c+4,col_c+2))
    vload(Cv[6], &C(row_c,col_c+3))
    vload(Cv[7], &C(row_c+4,col_c+3))
    vload(Cv[8], &C(row_c,col_c+4))
    vload(Cv[9], &C(row_c+4,col_c+4))
    vload(Cv[10], &C(row_c,col_c+5))
    vload(Cv[11], &C(row_c+4,col_c+5))
    vload(Cv[12], &C(row_c,col_c+6))
    vload(Cv[13], &C(row_c+4,col_c+6))
    vload(Cv[14], &C(row_c,col_c+7))
    vload(Cv[15], &C(row_c+4,col_c+7))
    
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
    simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
    simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
    simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

    simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
    simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
    simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
    simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

    simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
    simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
    simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
    simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

    vstore(&C(row_c,col_c), Cres[0])
    vstore(&C(row_c+4,col_c), Cres[1])
    vstore(&C(row_c,col_c+1), Cres[2])
    vstore(&C(row_c+4,col_c+1), Cres[3])
    vstore(&C(row_c,col_c+2), Cres[4])
    vstore(&C(row_c+4,col_c+2), Cres[5])
    vstore(&C(row_c,col_c+3), Cres[6])
    vstore(&C(row_c+4,col_c+3), Cres[7])
    vstore(&C(row_c,col_c+4), Cres[8])
    vstore(&C(row_c+4,col_c+4), Cres[9])
    vstore(&C(row_c,col_c+5), Cres[10])
    vstore(&C(row_c+4,col_c+5), Cres[11])
    vstore(&C(row_c,col_c+6), Cres[12])
    vstore(&C(row_c+4,col_c+6), Cres[13])
    vstore(&C(row_c,col_c+7), Cres[14])
    vstore(&C(row_c+4,col_c+7), Cres[15])
}