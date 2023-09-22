all: build
build: async_api.bin
PHONY: all build clean

NVCC := nvcc
NVCCFLAGS := -std=c++11 -m64 -arch=sm_89 --ptxas-options=-v -g -G

async_api.o: async_api.cu
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

async_api.bin: async_api.o
	@$(NVCC) $(NVCCFLAGS) $< -o $@