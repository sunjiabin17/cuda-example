all: build
build: c++11_cuda
PHONY: all build clean

NVCC := nvcc
NVCCFLAGS := -std=c++11 -m64

# @表示不显示命令本身
c++11_cuda.o: c++11_cuda.cu
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

c++11_cuda: c++11_cuda.o
	@$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf *.o c++11_cuda