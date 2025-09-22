#include <cuda_runtime.h>
#include <inttypes.h>
#include <iostream>
#include <chrono>

constexpr long long int SIZE = (1LL * 1024 * 1024 * 1024);
constexpr int THREADS_PER_BLOCK = 256;
constexpr int REPS = 10;
constexpr int N = 10;
constexpr int BLOCKS = (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

enum MemoryType {PINNED, NORMAL};

__global__ void addOneKernel(unsigned char* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

MemoryType memoryTypeUsed = PINNED;


void version1() {
    unsigned char *h_data = nullptr;
    unsigned char *d_data = nullptr;
    cudaMalloc((void**) &d_data, SIZE);

	if(NORMAL == memoryTypeUsed){
        h_data = (unsigned char*)malloc(SIZE);
	} else {
        cudaHostAlloc((void**) &h_data, SIZE, cudaHostAllocDefault);
    }

    memset(h_data, 0, SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    for(auto rep = 0; rep < REPS; rep++) {
            cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);

            for (int i = 0; i < N; ++i) {
            	addOneKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, SIZE);
            }

            cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
    }

	auto end = std::chrono::high_resolution_clock::now();

    cudaFree(d_data);
    if (NORMAL == memoryTypeUsed) free(h_data); else cudaFreeHost(h_data);

	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Version 1 CPU time: " << elapsed.count() << "s\n";
}

void version2() {
    unsigned char *h_data = nullptr;
    unsigned char *d_data = nullptr;
    cudaMalloc((void**) &d_data, SIZE);

    if(NORMAL == memoryTypeUsed){
        h_data = (unsigned char*)malloc(SIZE);
    } else {
        cudaHostAlloc((void**) &h_data, SIZE, cudaHostAllocDefault);
    }

	auto start = std::chrono::high_resolution_clock::now();

	for(auto rep = 0; rep < REPS; rep++) {
            	for (int i = 0; i < N; ++i) {
            		cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
            		addOneKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, SIZE);
            		cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
            	}
            }

	auto end = std::chrono::high_resolution_clock::now();


	cudaFree(d_data);
    if (NORMAL == memoryTypeUsed) free(h_data); else cudaFreeHost(h_data);
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Version 2 CPU time: " << elapsed.count() << "s\n";
}

int main() {

    if(NORMAL==memoryTypeUsed) {
std::cout << "Testing NORMAL memory mode\n";
        }else {

std::cout << "Testing PINNED memory mode\n";
        }

    for(int i = 0; i < 5; i++) {
        version1();
        version2();
    }
    return 0;
}
