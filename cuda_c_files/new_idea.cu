#include <cuda_runtime.h>
#include <inttypes.h>
#include <iostream>
#include <chrono>

#define SIZE (1LL * 1024 * 1024 * 1024)
#define THREADS_PER_BLOCK 256

__global__ void addOneKernel(unsigned char* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

void version1(int N) {
	auto REPS = 10;
	auto start = std::chrono::high_resolution_clock::now();
	for(auto rep = 0; rep < REPS; rep++) {
		uint64_t dataSize = SIZE;
		unsigned char *h_data = (unsigned char*)malloc(dataSize);
		unsigned char *d_data;

		memset(h_data, 0, dataSize);
		cudaMalloc(&d_data, dataSize);

		//cudaEvent_t startEvent, stopEvent;
		//float elapsedTime;
		//cudaEventCreate(&startEvent);
		//cudaEventCreate(&stopEvent);

		//cudaEventRecord(startEvent, 0);

		cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);

		int blocks = (dataSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		for (int i = 0; i < N; ++i) {
			addOneKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, dataSize);
		}

		cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost);

		//cudaEventRecord(stopEvent, 0);
		//cudaEventSynchronize(stopEvent);
		//cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

		//std::cout << "Version 1 GPU event time: " << elapsedTime / 1000.0f << "s\n";

		//cudaEventDestroy(startEvent);
		//cudaEventDestroy(stopEvent);
		cudaFree(d_data);
		free(h_data);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Version 1 CPU time: " << elapsed.count() << "s\n";

}

void version2(int N) {
	auto REPS = 10;
	auto start = std::chrono::high_resolution_clock::now();
	for(auto rep = 0; rep < REPS; rep++) {
		uint64_t dataSize = SIZE;
		unsigned char *h_data = (unsigned char*)malloc(dataSize);
		unsigned char *d_data;

		memset(h_data, 0, dataSize);
		cudaMalloc(&d_data, dataSize);

		//cudaEvent_t startEvent, stopEvent;
		//float elapsedTime;
		//cudaEventCreate(&startEvent);
		//cudaEventCreate(&stopEvent);

		//auto start = std::chrono::high_resolution_clock::now();
		//cudaEventRecord(startEvent, 0);

		int blocks = (dataSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		for (int i = 0; i < N; ++i) {
			cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
			addOneKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, dataSize);
			cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost);
		}

		//cudaEventRecord(stopEvent, 0);
		//cudaEventSynchronize(stopEvent);
		//cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		//auto end = std::chrono::high_resolution_clock::now();

		//std::chrono::duration<double> elapsed = end - start;
		//std::cout << "Version 2 CPU time: " << elapsed.count() << "s\n";
		//std::cout << "Version 2 GPU event time: " << elapsedTime / 1000.0f << "s\n";

		//cudaEventDestroy(startEvent);
		//cudaEventDestroy(stopEvent);
		cudaFree(d_data);
		free(h_data);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Version 2 CPU time: " << elapsed.count() << "s\n";
}

int main() {
    int N = 10; 
    version1(N);
    version2(N);
    return 0;
}
