#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void read_coalesced_write_stride_matrix_transpose(float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        output[x * height + y] = input[y * width + x];
    }
}

__global__ void read_stride_write_coalesced_matrix_transpose(float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        output[y * width + x] = input[x * height + y];
    }
}

__global__ void copy_row(float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        output[y * width + x] = input[y * width + x];
    }
}

__global__ void copy_column(float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        output[x * height + y] = input[x * height + y];
    }
}

__global__ void transposeUnroll4Row(float *input, float *output, int width, int height)
{
    unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    
    unsigned int ti = iy * width + ix; // access in rows
    unsigned int to = ix * height + iy; // access in columns

    if (ix + 3 * blockDim.x < width && iy < height)
    {
        output[to] = input[ti];
        output[to + height * blockDim.x] = input[ti + blockDim.x];
        output[to + height * 2 * blockDim.x] = input[ti + 2 * blockDim.x];
        output[to + height * 3 * blockDim.x] = input[ti + 3 * blockDim.x];
    }
}

__global__ void transpose_diagonal_row(float* input, float* output, int width, int height)
{
    int blk_x = blockIdx.x;
    int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;

    int ix = blockIdx.x * blk_x + threadIdx.x;
    int iy = blockIdx.y * blk_y + threadIdx.y;

    if (ix < width && iy < height)
    {
        output[ix * height + iy] = input[iy * width + ix];
    }
}

int main(int argc, char **argv)
{
    int width = 1 << 10;
    int height = 1 << 10;

    int blockx = 128;
    int blocky = 8;

    int mat_size = width * height;
    int mat_byte_size = mat_size * sizeof(float);

    float *h_matrix, *h_output, *h_ref;
    h_matrix = (float *)malloc(mat_byte_size);
    h_output = (float *)malloc(mat_byte_size);
    h_ref = (float *)malloc(mat_byte_size);

    // random initialization of input matrix
    for (int i = 0; i < mat_size; i++)
    {
        h_matrix[i] = (float)rand() / RAND_MAX;
    }

    // transpose matrix on cpu
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            h_ref[ix * height + iy] = h_matrix[iy * width + ix];
        }
    }

    dim3 grid(blockx, blocky);
    dim3 block((width + blockx - 1) / blockx, (height + blocky - 1) / blocky);

    float *d_matrix, *d_output;
    cudaMalloc((void **)&d_matrix, mat_byte_size);
    cudaMalloc((void **)&d_output, mat_byte_size);

    cudaMemcpy(d_matrix, h_matrix, mat_byte_size, cudaMemcpyHostToDevice);

    int iKernel = 0;
    if (argc > 1)
    {
        iKernel = atoi(argv[1]);
    }
    // kernel pointer and descriptor
    void (*kernel)(float *, float *, int, int);
    char *kernelName;

    // set up kernel
    switch (iKernel)
    {
    case 0:
        kernel = &copy_row;
        kernelName = "CopyRow       ";
        break;

    case 1:
        kernel = &copy_column;
        kernelName = "CopyCol       ";
        break;

    case 2:
        kernel = &read_coalesced_write_stride_matrix_transpose;
        kernelName = "read_coalesced_write_stride_matrix_transpose      ";
        break;

    case 3:
        kernel = &read_stride_write_coalesced_matrix_transpose;
        kernelName = "read_stride_write_coalesced_matrix_transpose      ";
        break;
    
    case 4:
        kernel = &transposeUnroll4Row;
        kernelName = "transposeUnroll4Row      ";
        break;

    case 5:
        kernel = &transpose_diagonal_row;
        kernelName = "transpose_diagonal_row      ";
        break;
    }

    // run kernel
    kernel<<<grid, block>>>(d_matrix, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, mat_byte_size, cudaMemcpyDeviceToHost);

    // check result
    bool bResult = true;
    for (int i = 0; i < mat_size; ++i)
    {
        if (h_ref[i] != h_output[i])
        {
            bResult = false;
            break;
        }
    }
    if (bResult)
    {
        printf("%s: PASS\n", kernelName);
    }
    else
    {
        printf("%s: FAIL\n", kernelName);
    }

    cudaFree(d_matrix);
    cudaFree(d_output);
    free(h_matrix);
    free(h_output);

    return 0;
}