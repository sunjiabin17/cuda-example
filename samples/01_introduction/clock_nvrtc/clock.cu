// #include <cuda_runtime.h>

// #include <iostream>

// 找到长度为NUM_THREADS的数组的最小值
// 一共找NUM_BLOCKS次，求平均耗时
extern "C" __global__ void timeReduction(const float* input, float* output,
                clock_t* timer) {
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) {
        timer[bid] = clock();
    }
    
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // find minimum
    for (int i = blockDim.x; i > 0; i >>= 1) {
        __syncthreads();
        if (tid < i) {
            float f0 = shared[tid];
            float f1 = shared[tid + i];

            if (f0 < f1) {
                shared[tid] = f0;
            }
        }
    }

    if (tid == 0) {
        output[bid] = shared[0];
    }
    __syncthreads();
    
    if (tid == 0) {
        timer[bid + gridDim.x] = clock();
    }
}
