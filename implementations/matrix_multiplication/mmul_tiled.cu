#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

#define N 1024
#define SHARED_SIZE 1024


__global__ void matrixMultiplication(int *a, int *b, int *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int a_shared[SHARED_SIZE];
    __shared__ int b_shared[SHARED_SIZE];

    int c_value = 0;

    // 把整个矩阵分成多个小块，每个小块的大小为blockDim.x * blockDim.y （32*32）
    for (int i = 0; i < n; i += blockDim.x) {
        // 把a矩阵中按先row后col的顺序划分每个小块，并加载到共享内存中
        // |1 2| |3 4|
        // |5 6| |7 8|
        // -----------
        // |9 0| |1 2|
        // |3 4| |5 6|
        // a ===> a_shared
        // |1 2|  |3 4|  |9 0|  |1 2|
        // |5 6|  |7 8|  |3 4|  |5 6|
        
        // row * N: 表示当前小块的第一行在a矩阵中的row偏移量
        // i: 表示当前小块的第一列在a矩阵中的col偏移量
        // threadIdx.x: 表示当前线程在小块中的col偏移量
        a_shared[threadIdx.y * blockDim.x + threadIdx.x] = a[row * n + i + threadIdx.x];
        

        // 把b矩阵中按先col后row的顺序划分每个小块，并加载到共享内存中
        // |1 2| |3 4|
        // |5 6| |7 8|
        // -----------
        // |9 0| |1 2|
        // |3 4| |5 6|
        // b ===> b_shared
        // |1 2|  |9 0|  |3 4|  |1 2|
        // |5 6|  |3 4|  |7 8|  |5 6|

        // shared memory中的b矩阵是按先col后row的顺序存储的  (例如b_shared中的第二个小块对应b矩阵中的blockIdx(1, 0)的位置)
        // i * N: 表示当前小块的第一行在b矩阵中的col偏移量(blockDim.y的整数倍)
        // threadIdx.y * N: 表示当前小块的第一行在b矩阵中的row偏移量 (blockDim.x的整数倍)
        // col: 表示当前线程在当前行中的row偏移量
        
        // 举例计算b[3][1]的值, blockIdx(1, 0)
        // i = 2, N = 4, threadIdx.y = 1, row = 3, col = 1
        // i * N = 8, threadIdx.y * N = 4, col = 1
        // b[3][1] = b[8 + 4 + 1] = b[13] = 4
        // b_shared[1][1] = b_shared[3] = 4
        b_shared[threadIdx.y * blockDim.x + threadIdx.x] = b[i * n + threadIdx.y * n + col];

        __syncthreads();

        // 计算当前小块的结果
        for (int j = 0; j < blockDim.x; ++j) {
            c_value += a_shared[threadIdx.y * blockDim.x + j] * b_shared[j * blockDim.x + threadIdx.x];
        }
    
        __syncthreads();
    }

    c[row * n + col] = c_value;
}


// matrix multiply on the CPU
void matrixMultiplicationCPU(const vector<int>& a, const vector<int>& b, vector<int>& c, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            c[row * n + col] = 0;
            for (int i = 0; i < n; ++i) {
                c[row * n + col] += a[row * n + i] * b[i * n + col];
            }
        }
    }
}

// verify that the GPU result matches the CPU result
bool verifyResult(const vector<int>& a, const vector<int>& b, int n) {
    for (int i = 0; i < n * n; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}


int main() {
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);
    vector<int> ref_c(N * N); 

    size_t bytes = N * N * sizeof(int);

    // initialize the matrices
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    int* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    matrixMultiplication<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    matrixMultiplicationCPU(h_a, h_b, ref_c, N);

    if (verifyResult(h_c, ref_c, N)) {
        cout << "the result is correct" << endl;
    } else {
        cout << "the result is incorrect" << endl;
    }

    // // print some value randomly
    // for (int i = 0; i < 10; ++i) {
    //     int row = rand() % N;
    //     int col = rand() % N;
    //     cout << "h_c[" << row << "][" << col << "] = " << h_c[row * N + col] << endl;
    //     cout << "ref_c[" << row << "][" << col << "] = " << ref_c[row * N + col] << endl;
    // }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}