#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void print_arrays_toafile(int*a, int size, char* name)
{
    std::ofstream file(name);

    if (file.is_open())
    {
        for (int i = 0; i < size; i++) {
            file << i << " - " << a[i] << "\n";
        }
        file.close();
    }
}

void print_arrays_toafile_side_by_side(int*a, int*b, int size, char* name)
{
    std::ofstream file(name);

    if (file.is_open())
    {
        for (int i = 0; i < size; i++) {
            file << i << " - " << a[i] << " - " << b[i] << "\n";
        }
        file.close();
    }
}

//compare arrays
void compare_arrays(int * a, int * b, int size)
{
    for (int  i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Arrays are different \n");
            printf("%d - %d | %d \n", i, a[i], b[i]);
            //return;
        }
    }
    printf("Arrays are same \n");
}



#define BLOCK_SIZE 1024

void inclusive_scan_cpu(int *input, int *output, int size)
{
	output[0] = input[0];

	for (int i = 1; i < size; i++)
	{
		output[i] = output[i - 1] + input[i];
	}
}

__global__ void naive_inclusive_scan_single_block(int *input, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		for (int stride = 1; stride <= tid; stride *= 2)
		{
			input[gid] += input[gid - stride];
			__syncthreads();
		}
	}
}

__global__ void efficient_exclusive_scan_single_block(int *input, int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		// reduction phase
		for (int stride = 1; stride < blockDim.x; stride *= 2) {
			int index = (tid + 1) * stride * 2 - 1;	// odd index
			if (index < blockDim.x) {
				input[index] += input[index - stride];
			}
			__syncthreads();
		}

		// set root value to 0
		if (tid == 0) {
			input[blockDim.x - 1] = 0;
		}

		int temp = 0;

		// down-sweep phase
		for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
			int index = (tid + 1) * stride * 2 - 1;
			if (index < blockDim.x) {
				temp = input[index - stride];	// assign left child value to temp
				input[index - stride] = input[index];	// assign parent value to left child
				input[index] += temp;	// add left child value to parent value
			}
			__syncthreads();
		}
	}
}


__global__ void efficient_inclusive_scan_single_block(int *input,int size)
{
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// shared memory optimization
	__shared__ int shared_block[BLOCK_SIZE];

	if (gid < size)
	{	
		// copy data to shared memory
		shared_block[tid] = input[gid];
		__syncthreads();

		// reduction phase (same as exclusive scan)
		for (int stride = 1; stride < blockDim.x; stride *= 2) {
			int index = (tid + 1) * stride * 2 - 1;	// odd index
			if (index < blockDim.x) {
				shared_block[index] += shared_block[index - stride];
			}
			__syncthreads();
		}

		// down-sweep phase
		for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
			int index = (tid + 1) * stride * 2 - 1;
			if (index + stride < blockDim.x) {
				shared_block[index + stride] += shared_block[index];
			}
			__syncthreads();
		}

		// copy data back to global memory
		input[gid] = shared_block[tid];
	}
}

__global__ void inclusive_prescan(int *input, int *aux, int size) {
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// shared memory optimization
	__shared__ int shared_block[BLOCK_SIZE];

	if (gid < size)
	{	
		// copy data to shared memory
		shared_block[tid] = input[gid];
		__syncthreads();

		// reduction phase
		for (int stride = 1; stride < blockDim.x; stride *= 2) {
			int index = (tid + 1) * stride * 2 - 1;	// odd index
			if (index < blockDim.x) {
				shared_block[index] += shared_block[index - stride];
			}
			__syncthreads();
		}

		// down-sweep phase
		for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
			int index = (tid + 1) * stride * 2 - 1;
			if (index + stride < blockDim.x) {
				shared_block[index + stride] += shared_block[index];
			}
			__syncthreads();
		}

		// copy data back to global memory
		input[gid] = shared_block[tid];
		
		if (tid == blockDim.x - 1) {
			aux[blockIdx.x] = shared_block[tid];
		}
	}
}

__global__ void inclusive_postscan(int *input, int *aux, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size and blockIdx.x > 0) {
		input[gid] += aux[blockIdx.x-1];
	}
}

int main(int argc, char**argv)
{
	printf("Scan algorithm execution starterd \n");

	int input_size = 1 << 10;
	
	if (argc > 1)
	{
		input_size = 1 << atoi(argv[1]);
	}
	
	const int byte_size = sizeof(int) * input_size;

	int * h_input, *h_output, *h_ref, *h_aux;

	clock_t cpu_start, cpu_end, gpu_start, gpu_end;

	h_input = (int*)malloc(byte_size);
	h_output = (int*)malloc(byte_size);
	h_ref = (int*)malloc(byte_size);

	// randomly initialize the input data
	for (int i = 0; i < input_size; i++)
	{
		h_input[i] = (int)(rand() & 0xFF);
	}

	cpu_start = clock();
	inclusive_scan_cpu(h_input, h_output, input_size);
	cpu_end = clock();

	int *d_input, *d_aux;
	cudaMalloc((void**)&d_input, byte_size);

	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE);
	dim3 grid(input_size/ block.x);

	int aux_byte_size = grid.x * sizeof(int);
	cudaMalloc((void**)&d_aux , aux_byte_size);

	h_aux = (int*)malloc(aux_byte_size);
	
	inclusive_prescan << <grid, block >> > (d_input, d_aux, input_size);
	efficient_inclusive_scan_single_block << <1, 1024 >> > (d_aux, grid.x);
	inclusive_postscan << <grid, block >> > (d_input, d_aux, input_size);

	cudaDeviceSynchronize();

	cudaMemcpy(h_aux, d_aux, aux_byte_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ref, d_input, byte_size, cudaMemcpyDeviceToHost);

	compare_arrays(h_ref, h_output, input_size);

	cudaDeviceReset();
	return 0;
}