#include <cuda_runtime.h>
#include <inttypes.h>
#include <iostream>
#include <chrono>

constexpr long long int SIZE = (1LL * 1024 * 1024 * 1024);
constexpr int THREADS_PER_BLOCK = 256;
constexpr int REPS = 10;
constexpr int N = 10;
constexpr int BLOCKS = (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
double kernel_sum = 0.0;
double host_to_device_transfer_sum = 0.0;
double device_to_host_transfer_sum = 0.0;

enum MemoryType {PINNED, NORMAL};

__global__ void addOneKernel(unsigned char* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}

void clear_timing() {
    kernel_sum = 0.0;
    host_to_device_transfer_sum = 0.0;
    device_to_host_transfer_sum = 0.0;
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
        auto t_start1 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
        auto t_end1 = std::chrono::high_resolution_clock::now();

        auto k_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            addOneKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, SIZE);
        }
        auto k_end = std::chrono::high_resolution_clock::now();

        auto t_start2 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
        auto t_end2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> t_time1 = t_end1 - t_start1;
        std::chrono::duration<double> t_time2 = t_end2 - t_start2;
        std::chrono::duration<double> k_time = k_end - k_start;

        kernel_sum += k_time.count();
        host_to_device_transfer_sum += t_time1.count();
        device_to_host_transfer_sum += t_time2.count();
    }
	auto end = std::chrono::high_resolution_clock::now();

    cudaFree(d_data);
    if (NORMAL == memoryTypeUsed) free(h_data); else cudaFreeHost(h_data);

	std::chrono::duration<double> elapsed = end - start;

	std::printf("Kernel sum: %.3f s, kernel avg: %.3f s\n", kernel_sum, kernel_sum / REPS);
	std::printf("H->D sum: %.3f s, H->D avg: %.3f s\n", host_to_device_transfer_sum, host_to_device_transfer_sum / REPS);
	std::printf("D->H sum: %.3f s, D->H avg: %.3f s\n", device_to_host_transfer_sum, device_to_host_transfer_sum / REPS);
	std::printf("Version 1 CPU time: %.3f s\n", elapsed.count());
	clear_timing();
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
	            auto k_start = std::chrono::high_resolution_clock::now();
            	for (int i = 0; i < N; ++i) {
            	    auto t_start1 = std::chrono::high_resolution_clock::now();
            		cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice);
            		auto t_end1 = std::chrono::high_resolution_clock::now();

            		addOneKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, SIZE);

            		auto t_start2 = std::chrono::high_resolution_clock::now();
            		cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost);
            		auto t_end2 = std::chrono::high_resolution_clock::now();

            		std::chrono::duration<double> t_time1 = t_end1 - t_start1;
                    std::chrono::duration<double> t_time2 = t_end2 - t_start2;

                    host_to_device_transfer_sum += t_time1.count();
                    device_to_host_transfer_sum += t_time2.count();
            	}
            	auto k_end = std::chrono::high_resolution_clock::now();
            	std::chrono::duration<double> k_time = k_end - k_start;
            	kernel_sum += k_time.count();
            }

	auto end = std::chrono::high_resolution_clock::now();


	cudaFree(d_data);
    if (NORMAL == memoryTypeUsed) free(h_data); else cudaFreeHost(h_data);
	std::chrono::duration<double> elapsed = end - start;

	std::printf("Kernel sum: %.3f s, kernel avg: %.3f s\n", kernel_sum, kernel_sum / REPS);
    std::printf("H->D sum: %.3f s, H->D avg: %.3f s\n", host_to_device_transfer_sum, host_to_device_transfer_sum / REPS);
    std::printf("D->H sum: %.3f s, D->H avg: %.3f s\n", device_to_host_transfer_sum, device_to_host_transfer_sum / REPS);
	std::cout << "Version 2 CPU time: " << elapsed.count() << "s\n";
	clear_timing();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
            std::cout << "Usage: " << argv[0] << " <1|2>\n";
            return 1;
        }

    std::string arg = argv[1];

        if (arg == "1") {
            std::cout << "Testing NORMAL memory mode\n";
            memoryTypeUsed = NORMAL;
        } else if (arg == "2") {
            std::cout << "Testing PINNED memory mode\n";
            memoryTypeUsed = PINNED;
        } else {
            std::cout << "Invalid option: " << arg << "\n";
        }

    for(int i = 0; i < 5; i++) {
        version1();
        version2();
    }
    return 0;
}
