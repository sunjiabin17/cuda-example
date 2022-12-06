#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void blocking_nonblocking_test1()
{
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid == 0)
	{
		for (size_t i = 0; i < 10000; i++)
		{
			printf("kernel 1 \n");
		}
	}
}

int main(int argc, char **argv)
{
	int size = 1 << 15;

	cudaStream_t stm1, stm2, stm3;
	cudaStreamCreateWithFlags(&stm1, cudaStreamNonBlocking);
	cudaStreamCreate(&stm2);
	cudaStreamCreateWithFlags(&stm3, cudaStreamNonBlocking);

	dim3 block(128);
	dim3 grid(size / block.x);

	blocking_nonblocking_test1<<<grid, block, 0, stm1>>>();
	blocking_nonblocking_test1<<<grid, block>>>();
	blocking_nonblocking_test1<<<grid, block, 0, stm3>>>();

	cudaStreamDestroy(stm1);
	cudaStreamDestroy(stm2);
	cudaStreamDestroy(stm3);
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}