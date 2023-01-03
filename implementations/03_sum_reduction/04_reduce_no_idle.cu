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
// note: 256
constexpr int SHARED_SIZE = 256;

// 解决threads idle的问题
__global__ void reduce(int *input, int *output) {
    __shared__ int shared[SHARED_SIZE];
    
    int gid = blockIdx.x * (2*blockDim.x) + threadIdx.x;
    
    // thread 0: 计算input[0] + input[256] -> shared[0]
    // thread 1: 计算input[1] + input[257] -> shared[1]
    // ...
    // thread 255: 计算input[255] + input[511] -> shared[255]
    // 一个block处理512个元素
    shared[threadIdx.x] = input[gid] + input[gid + blockDim.x];
    __syncthreads();

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

    // note: BLOCKS/2
    reduce<<<BLOCKS/2, THREADS>>>(d_input, d_output);

    reduce<<<1, BLOCKS>>>(d_output, d_output);

    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);


    cout << "Host sum: " << h_sum << endl;
    cout << "Device sum: " << h_output[0] << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}