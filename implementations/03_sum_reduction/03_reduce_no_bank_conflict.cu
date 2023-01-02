#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using std::cout;
using std::endl;
using std::vector;
using std::accumulate;
using std::generate;

constexpr int THREADS = 256;
constexpr int SHARED_SIZE = 256;

// 解决bank conflict
__global__ void reduce(int *input, int *output) {
    __shared__ int shared[SHARED_SIZE];
    
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[threadIdx.x] = input[gid];
    __syncthreads();

    // blockDim.x = 256
    // 第一次迭代: i = 128
    // thread 0: 计算shared[0] + shared[128] -> shared[0]
    // thread 1: 计算shared[1] + shared[129] -> shared[1]
    // ...
    // warp 0: 读取shared memory 中连续的[0-31] 和 [128-159]
    // warp 1: 读取shared memory 中连续的[32-63] 和 [160-191]
    // ...
    // 不存在bank conflict
    // 第二次迭代: i = 64
    // thread 0: 计算shared[0] + shared[64] -> shared[0]
    // thread 1: 计算shared[1] + shared[65] -> shared[1]
    // ...
    // warp 0: 读取shared memory 中连续的[0-31] 和 [64-95]
    // warp 1: 读取shared memory 中连续的[32-63] 和 [96-127]
    // ...
    // 不存在bank conflict
    // 等等...
    for(unsigned int i=blockDim.x/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared[threadIdx.x];
    }
}

int main() {
    constexpr int N = 1 << 16;
    int size = N * sizeof(int);
    vector<int> h_input(N);
    vector<int> h_output(N);

    generate(h_input.begin(), h_input.end(), []() { return rand() % 100; });
    int h_sum = accumulate(h_input.begin(), h_input.end(), 0);

    int *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);

    constexpr int BLOCKS = N / THREADS;

    reduce<<<BLOCKS, THREADS>>>(d_input, d_output);

    reduce<<<1, BLOCKS>>>(d_output, d_output);

    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);


    cout << "Host sum: " << h_sum << endl;
    cout << "Device sum: " << h_output[0] << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}