#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace cooperative_groups;

using std::cout;
using std::endl;
using std::vector;
using std::accumulate;
using std::generate;


__device__ int thread_sum(int *input, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    // for (int i = tid; i < n / 4; i += blockDim.x * gridDim.x) {
    //     int4 input4 = reinterpret_cast<int4*>(input)[i];
    //     sum += input4.x + input4.y + input4.z + input4.w;
    // }
    if (tid < n/4) {
        int4 input4 = reinterpret_cast<int4*>(input)[tid];
        sum += input4.x + input4.y + input4.z + input4.w;
    }
    return sum;
}

// Reduces a thread group to a single element
__device__ int reduce_sum_thread_group(const thread_group& g, int *shared, int val) {
    int lane = g.thread_rank();
    shared[lane] = val;
    g.sync();

    for (int i = g.size() / 2; i > 0; i >>= 1) {
        if (lane < i) {
            shared[lane] += shared[lane + i];
        }
        g.sync();
    }
    return shared[lane];
}

__global__ void reduce(int *input, int *output, int n) {
    int my_sum = thread_sum(input, n);
    extern __shared__ int shared[];
    auto g = this_thread_block();
    int block_sum = reduce_sum_thread_group(g, shared, my_sum);
    // printf("g.thread_rank()=%d, block_sum=%d\n", g.thread_rank(), block_sum);
    if (g.thread_rank() == 0) {
        // printf("g.size()=%d, threadIdx=%d, block_sum=%d\n", g.size(), threadIdx.x + blockIdx.x * blockDim.x, block_sum);
        atomicAdd(output, block_sum);
    }
}

int main() {
    constexpr int N = 1 << 20;
    size_t bytes = N * sizeof(int);
    int *output, *input;
 
    cudaMallocManaged(&output, sizeof(int));
    cudaMallocManaged(&input, bytes);
    std::fill(input, input + N, 1);

    int h_sum = accumulate(input, input + N, 0);

    
    constexpr int THREADS = 256;
    constexpr int BLOCKS = (N + THREADS - 1) / THREADS;

    reduce<<<BLOCKS/4, THREADS, THREADS * sizeof(int)>>>(input, output, N);
    cudaDeviceSynchronize();


    cout << "Host sum: " << h_sum << endl;
    cout << "Device sum: " << output[0] << endl;

    cudaFree(output);
    cudaFree(input);

    return 0;
}