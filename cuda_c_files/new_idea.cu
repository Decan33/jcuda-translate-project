#include <cuda_runtime.h>
#include <inttypes.h>
#include <iostream>
#include <chrono>

constexpr long long int SIZE = (1LL * 1024 * 1024 * 1024);
constexpr int THREADS_PER_BLOCK = 256;
constexpr int REPS = 10;
constexpr int N = 10;

enum MemoryType {PINNED, NORMAL};

__global__ void addOneKernel(unsigned char* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

MemoryType memoryTypeUsed = PINNED;


void version1() {
	auto start = std::chrono::high_resolution_clock::now();
	if(NORMAL == memoryTypeUsed){
	    for(auto rep = 0; rep < REPS; rep++) {
        		unsigned char *h_data = (unsigned char*)malloc(SIZE);
        		unsigned char *d_data;

        		memset(h_data, 0, SIZE);
        		cudaMalloc(&d_data, SIZE);

        		cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);

        		int blocks = (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        		for (int i = 0; i < N; ++i) {
        			addOneKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, SIZE);
        		}

        		cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);

        		cudaFree(d_data);
        		free(h_data);
        	}
	} else {
	    for(auto rep = 0; rep < REPS; rep++) {
        	unsigned char *h_data = (unsigned char*)malloc(SIZE);
        	unsigned char *d_data;

        	memset(h_data, 0, SIZE);
        	cudaHostAlloc(&d_data, SIZE, cudaHostAllocDefault);

        	cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);

        	int blocks = (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        	for (int i = 0; i < N; ++i) {
        		addOneKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, SIZE);
        	}

        	cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);

        	cudaFreeHost(d_data);
        	free(h_data);
        }
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Version 1 CPU time: " << elapsed.count() << "s\n";
}

void version2() {
	auto start = std::chrono::high_resolution_clock::now();
	if(NORMAL==memoryTypeUsed) {
	    for(auto rep = 0; rep < REPS; rep++) {
        	unsigned char *h_data = (unsigned char*)malloc(SIZE);
        	unsigned char *d_data;

        	memset(h_data, 0, SIZE);
        	cudaMalloc(&d_data, SIZE);

        	int blocks = (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        	for (int i = 0; i < N; ++i) {
        		cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
        		addOneKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, SIZE);
        		cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
        	}
        	cudaFree(d_data);
        	free(h_data);
        }
	}else {
	    for(auto rep = 0; rep < REPS; rep++) {
        	unsigned char *h_data = (unsigned char*)malloc(SIZE);
        	unsigned char *d_data;

        	memset(h_data, 0, SIZE);
        	cudaHostAlloc(&d_data, SIZE, cudaHostAllocDefault);

        	int blocks = (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        	for (int i = 0; i < N; ++i) {
        		cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
        		addOneKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, SIZE);
        		cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
        	}
        	cudaFreeHost(d_data);
        	free(h_data);
        }
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Version 2 CPU time: " << elapsed.count() << "s\n";
}

int main() {
    for(int i = 0; i < 5; i++) {
        version1();
        version2();
    }
    return 0;
}
