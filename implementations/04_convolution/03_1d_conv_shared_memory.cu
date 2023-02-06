#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CONV_1D_KERNEL  7
#define THREADS         256

__constant__ int d_mask[CONV_1D_KERNEL];

__global__ void convolution_1d(int *input, int *output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = 0;
    const int n_padded = CONV_1D_KERNEL / 2 * 2;

    __shared__ int s_input[THREADS + n_padded];
    s_input[threadIdx.x] = input[tid];
    // 还有n_padded个数没有加载到共享内存的末尾
    // 方法一
    int offset = threadIdx.x + blockDim.x;
    if (offset < n_padded + blockDim.x) {
        s_input[offset] = input[tid + blockDim.x];
    }
    // 方法二
    // if (threadIdx.x >= blockDim.x - n_padded) {
    //     s_input[threadIdx.x + n_padded] = input[tid + n_padded];
    // }

    __syncthreads();

    for (int i = 0; i < CONV_1D_KERNEL; i++) {
        temp += s_input[threadIdx.x + i] * d_mask[i];
    }
    output[tid] = temp;
}

// Verify the result on the CPU
void verify_result(int *input, int *mask, int *output, int n) {
  int temp;
  for (int i = 0; i < n; i++) {
    temp = 0;
    for (int j = 0; j < CONV_1D_KERNEL; j++) {
      temp += input[i + j] * mask[j];
    }
    assert(temp == output[i]);
  }
}

int main() {
    int n = 1 << 20;
    int bytes_n = n * sizeof(int);
    int np = n + CONV_1D_KERNEL / 2 * 2;
    int bytes_np = np * sizeof(int);
    int bytes_m = CONV_1D_KERNEL * sizeof(int);

    std::vector<int> h_input(np, 0);

    std::generate(begin(h_input) + CONV_1D_KERNEL/2, end(h_input) - CONV_1D_KERNEL/2, [](){ return rand() % 100; });

    std::vector<int> h_mask(CONV_1D_KERNEL);
    std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

    std::vector<int> h_output(n, 0);

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes_np);
    cudaMalloc(&d_output, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_input, h_input.data(), bytes_np, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask.data(), bytes_m);

    int GRID = (n + THREADS - 1) / THREADS;

    convolution_1d<<<GRID, THREADS>>>(d_input, d_output, n);

    cudaMemcpy(h_output.data(), d_output, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(h_input.data(), h_mask.data(), h_output.data(), n);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_output);
    cudaFree(d_input);

    return 0;
}