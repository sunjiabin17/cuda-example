#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CONV_2D_KERNEL  7
#define THREADS         16

__constant__ int d_mask[CONV_2D_KERNEL * CONV_2D_KERNEL];

__global__ void convolution_2d(int *input, int *output, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp = 0;

    int start_row = row - CONV_2D_KERNEL / 2;
    int start_col = col - CONV_2D_KERNEL / 2;

    for (int i = 0; i < CONV_2D_KERNEL; ++i) {
        for (int j = 0; j < CONV_2D_KERNEL; ++j) {
            if (start_row + i >= 0 && start_row + i < n && start_col + j >= 0 && start_col + j < n) {
                temp += input[(start_row + i) * n + (start_col + j)] * d_mask[i * CONV_2D_KERNEL + j];
            }
        }
    }
    output[row * n + col] = temp;
}

// Verify the result on the CPU
void verify_result(int *input, int *mask, int *output, int n) {
    int temp;
    int offset_row, offset_col;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp = 0;
            for (int k = 0; k < CONV_2D_KERNEL; k++) {
                offset_row = i + k - CONV_2D_KERNEL / 2;
                for (int l = 0; l < CONV_2D_KERNEL; l++) {
                    offset_col = j + l - CONV_2D_KERNEL / 2;
                    if (offset_row >= 0 && offset_row < n && offset_col >= 0 && offset_col < n) {
                        temp += input[offset_row * n + offset_col] * mask[k * CONV_2D_KERNEL + l];
                    }
                }
            }
            assert(temp == output[i * n + j]);
        }
    }
}

int main() {
    int n = 1 << 10;
    size_t bytes_n = n * n * sizeof(int);
    size_t bytes_m = CONV_2D_KERNEL * CONV_2D_KERNEL * sizeof(int);

    std::vector<int> h_input(n * n, 0);

    std::generate(begin(h_input), end(h_input), [](){ return rand() % 100; });

    std::vector<int> h_mask(CONV_2D_KERNEL * CONV_2D_KERNEL);
    std::generate(begin(h_mask), end(h_mask), [](){ return rand() % 10; });

    std::vector<int> h_output(n * n, 0);

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes_n);
    cudaMalloc(&d_output, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_input, h_input.data(), bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask.data(), bytes_m);

    dim3 blocks(THREADS, THREADS);
    dim3 grids((n + THREADS - 1) / THREADS, (n + THREADS - 1) / THREADS);

    convolution_2d<<<grids, blocks>>>(d_input, d_output, n);

    cudaMemcpy(h_output.data(), d_output, bytes_n, cudaMemcpyDeviceToHost);

    verify_result(h_input.data(), h_mask.data(), h_output.data(), n);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_output);
    cudaFree(d_input);

    return 0;
}