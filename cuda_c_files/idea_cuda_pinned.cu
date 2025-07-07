#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>
#include "data.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int CHUNK_SIZE = 250000000; // match Java

__constant__ float const_tmin;
__constant__ float const_delta;
__constant__ int   const_coefficients;

__constant__ float const_pi;
__constant__ float const_pi_squared;
__constant__ float const_T;
__constant__ float const_pi_over_T;
__constant__ float constant_result_coefficient;

__global__ void fourier(float* results)
{
    __shared__ float shared_coefficients[MAX_COEFFICIENTS];

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx = threadIdx.x;

    for (auto k = idx; k < const_coefficients; k += blockDim.x)
    {
        auto denominator = 4.0f * (k + 1) * (k + 1) - 4.0f * (k + 1) + 1.0f;
        shared_coefficients[k] = 1.0f / denominator;
    }

    __syncthreads();

    if (tid >= gridDim.x * blockDim.x) return;

    auto t = const_tmin + tid * const_delta;
    auto sum = 0.0f;

    for (auto k = 1; k <= const_coefficients; ++k)
    {
        auto angle = (2 * k - 1) * const_pi_over_T * t;
        auto numerator = cosf(angle);
        sum += numerator * shared_coefficients[k - 1];
    }

    results[tid] = const_T * 0.5f - (constant_result_coefficient * sum);
}

void performColdRun(float tmin, float tmax, int length, int coefficients, float delta) {
    float* result_device;
    CUDA_CHECK(cudaMalloc((void**)&result_device, length * sizeof(float)));

    auto blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    CUDA_CHECK(cudaMemcpyToSymbol(const_tmin,   &tmin, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_delta,  &delta, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_coefficients, &coefficients, sizeof(int)));

    CUDA_CHECK(cudaMemcpyToSymbol(const_pi, &pi, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi_squared, &pi_squared, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_T, &T, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi_over_T, &pi_over_T, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(constant_result_coefficient, &result_coefficient, sizeof(float)));

    fourier<<<blocks, THREADS_PER_BLOCK>>>(result_device);
    CUDA_CHECK(cudaDeviceSynchronize());

    int actualChunkSize = std::min(CHUNK_SIZE, length);
    float* result_host;
    CUDA_CHECK(cudaHostAlloc((void**)&result_host, actualChunkSize * sizeof(float), cudaHostAllocDefault));
    for (int offset = 0; offset < length; offset += actualChunkSize) {
        int thisChunk = std::min(actualChunkSize, length - offset);
        CUDA_CHECK(cudaMemcpy(result_host, result_device + offset, thisChunk * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFreeHost(result_host));
    CUDA_CHECK(cudaFree(result_device));
}

int main()
{
    long long requiredDeviceMemory = (long long)length * sizeof(float);
    long long requiredHostMemory = (long long)std::min(CHUNK_SIZE, length) * sizeof(float);
    printf("Memory requirements: Device=%.2f GB, Host chunk=%.2f MB\n",
        requiredDeviceMemory / (1024.0 * 1024.0 * 1024.0),
        requiredHostMemory / (1024.0 * 1024.0));
    printf("TESTING FOURIER USING PINNED MEMORY\n");
    printf("Performing cold run to warm up GPU...\n");
    performColdRun(tmin, tmax, length, coefficients, delta);
    printf("Cold run completed.\n\n");

    std::vector<double> prep_times, kernel_times, copy_times, delete_times;

    auto start_reps = std::chrono::high_resolution_clock::now();
    for (auto rep = 0; rep < NUM_REPS; ++rep) {
        auto prep_start = std::chrono::high_resolution_clock::now();
        
        float* result_device;
        CUDA_CHECK(cudaMalloc((void**)&result_device, length * sizeof(float)));

        auto blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        CUDA_CHECK(cudaMemcpyToSymbol(const_tmin,   &tmin, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(const_delta,  &delta, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(const_coefficients, &coefficients, sizeof(int)));

        CUDA_CHECK(cudaMemcpyToSymbol(const_pi, &pi, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(const_pi_squared, &pi_squared, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(const_T, &T, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(const_pi_over_T, &pi_over_T, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(constant_result_coefficient, &result_coefficient, sizeof(float)));

        auto prep_end = std::chrono::high_resolution_clock::now();
        prep_times.push_back(std::chrono::duration<double>(prep_end - prep_start).count());
        
        cudaEvent_t kernel_start, kernel_stop;
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));
        CUDA_CHECK(cudaEventRecord(kernel_start));
        
        fourier<<<blocks, THREADS_PER_BLOCK>>>(result_device);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));
        
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms / 1000.0);
        
        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
        
        auto copy_start = std::chrono::high_resolution_clock::now();
        int actualChunkSize = std::min(CHUNK_SIZE, length);
        float* result_host;
        CUDA_CHECK(cudaHostAlloc((void**)&result_host, actualChunkSize * sizeof(float), cudaHostAllocDefault));
        for (int offset = 0; offset < length; offset += actualChunkSize) {
            int thisChunk = std::min(actualChunkSize, length - offset);
            CUDA_CHECK(cudaMemcpy(result_host, result_device + offset, thisChunk * sizeof(float), cudaMemcpyDeviceToHost));
        }
        auto copy_end = std::chrono::high_resolution_clock::now();
        copy_times.push_back(std::chrono::duration<double>(copy_end - copy_start).count());
        
        auto delete_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaFreeHost(result_host));
        CUDA_CHECK(cudaFree(result_device));
        auto delete_end = std::chrono::high_resolution_clock::now();
        
        delete_times.push_back(std::chrono::duration<double>(delete_end - delete_start).count());
    }
    auto end_reps = std::chrono::high_resolution_clock::now();
    
    double prep_sum = 0, kernel_sum = 0, copy_sum = 0, delete_sum = 0;
    
    for (auto i = 0u; i < prep_times.size(); ++i) {
	    if(logReps) {
	        printf("Repetition %u:\n", i + 1);
        	printf("  Preparation time: %.6f s\n", prep_times[i]);
        	printf("  Kernel execution time: %.6f s\n", kernel_times[i]);
        	printf("  Data copy time: %.6f s\n", copy_times[i]);
        	printf("  Memory deletion time: %.6f s\n", delete_times[i]);
	    }
        prep_sum += prep_times[i];
        kernel_sum += kernel_times[i];
        copy_sum += copy_times[i];
        delete_sum += delete_times[i];
    }
    
    auto n = static_cast<double>(prep_times.size());
    double prep_avg = prep_sum / n;
    double kernel_avg = kernel_sum / n;
    double copy_avg = copy_sum / n;
    double delete_avg = delete_sum / n;
    double prep_var = 0, kernel_var = 0, copy_var = 0, delete_var = 0;
    
    for (auto i = 0u; i < prep_times.size(); ++i) {
        prep_var += (prep_times[i] - prep_avg) * (prep_times[i] - prep_avg);
        kernel_var += (kernel_times[i] - kernel_avg) * (kernel_times[i] - kernel_avg);
        copy_var += (copy_times[i] - copy_avg) * (copy_times[i] - copy_avg);
        delete_var += (delete_times[i] - delete_avg) * (delete_times[i] - delete_avg);
    }
    
    double prep_std = std::sqrt(prep_var / n);
    double kernel_std = std::sqrt(kernel_var / n);
    double copy_std = std::sqrt(copy_var / n);
    double delete_std = std::sqrt(delete_var / n);
    
    printf("\nAverages over %zu repetitions:\n", prep_times.size());
    printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prep_avg, prep_std);
    printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernel_avg, kernel_std);
    printf("  Avg data copy time: %.6f s (stddev: %.6f s)\n", copy_avg, copy_std);
    printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delete_avg, delete_std);
    printf("  Whole time taken for %d reps: %.6f s\n", NUM_REPS, std::chrono::duration<double>(end_reps - start_reps).count());
    printf("=========================\n\n");
    return 0;
} 