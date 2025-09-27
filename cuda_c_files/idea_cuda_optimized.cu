#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <chrono>
#include "data.h"

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

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;

    for (int k = idx; k < const_coefficients; k += blockDim.x)
    {
        float denominator = 4.0f * (k + 1) * (k + 1) - 4.0f * (k + 1) + 1.0f;
        shared_coefficients[k] = 1.0f / denominator;
    }

    __syncthreads();

    if (tid >= gridDim.x * blockDim.x) return;

    float t = const_tmin + tid * const_delta;
    float sum = 0.0f;

    for (auto k = 1; k <= const_coefficients; ++k)
    {
        float angle = (2 * k - 1) * const_pi_over_T * t;
        float numerator = cosf(angle);
        sum += numerator * shared_coefficients[k - 1];
    }

    results[tid] = const_T * 0.5f - (constant_result_coefficient * sum);
}

static float* result_device = nullptr;
static float* result_host = nullptr;
static cudaEvent_t kernel_start, kernel_stop;
static bool memory_initialized = false;

void initializeMemory() {
    if (memory_initialized) return;
    
    CUDA_CHECK(cudaMalloc((void**)&result_device, length * sizeof(float)));
    result_host = new float[length];
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    
    memory_initialized = true;
}

void cleanupMemory() {
    if (!memory_initialized) return;
    
    CUDA_CHECK(cudaFree(result_device));
    delete[] result_host;
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    
    memory_initialized = false;
}

void setAllConstants() {
    CUDA_CHECK(cudaMemcpyToSymbol(const_tmin, &tmin, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_delta, &delta, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_coefficients, &coefficients, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi, &pi, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi_squared, &pi_squared, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_T, &T, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi_over_T, &pi_over_T, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(constant_result_coefficient, &result_coefficient, sizeof(float)));
}

void performColdRun() {
    printf("Performing cold run to warm up GPU...\n");
    
    initializeMemory();
    setAllConstants();

    fourier<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(result_device);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Cold run completed.\n\n");
}

void runSingleTest(std::vector<double>& prep_times, std::vector<double>& kernel_times, 
                   std::vector<double>& copy_times, std::vector<double>& cleanup_times) {
    
    auto prep_start = std::chrono::high_resolution_clock::now();
    setAllConstants();
    auto prep_end = std::chrono::high_resolution_clock::now();
    prep_times.push_back(std::chrono::duration<double>(prep_end - prep_start).count());

    CUDA_CHECK(cudaEventRecord(kernel_start));
    fourier<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(result_device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));
    
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    kernel_times.push_back(kernel_ms / 1000.0);

    auto copy_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(result_host, result_device, length * sizeof(float), cudaMemcpyDeviceToHost));
    auto copy_end = std::chrono::high_resolution_clock::now();
    copy_times.push_back(std::chrono::duration<double>(copy_end - copy_start).count());

    auto cleanup_start = std::chrono::high_resolution_clock::now();
    auto cleanup_end = std::chrono::high_resolution_clock::now();
    cleanup_times.push_back(std::chrono::duration<double>(cleanup_end - cleanup_start).count());
}

void printResults(const std::vector<double>& prep_times, const std::vector<double>& kernel_times,
                  const std::vector<double>& copy_times, const std::vector<double>& cleanup_times,
                  double total_time) {
    
    if (logReps) {
        for (size_t i = 0; i < prep_times.size(); ++i) {
            printf("Repetition %zu:\n", i + 1);
            printf("  Preparation time: %.6f s\n", prep_times[i]);
            printf("  Kernel execution time: %.6f s\n", kernel_times[i]);
            printf("  Data copy time: %.6f s\n", copy_times[i]);
            printf("  Memory deletion time: %.6f s\n", cleanup_times[i]);
        }
    }

    double prep_sum = 0, kernel_sum = 0, copy_sum = 0, cleanup_sum = 0;
    for (size_t i = 0; i < prep_times.size(); ++i) {
        prep_sum += prep_times[i];
        kernel_sum += kernel_times[i];
        copy_sum += copy_times[i];
        cleanup_sum += cleanup_times[i];
    }

    const double n = static_cast<double>(prep_times.size());
    const double prep_avg = prep_sum / n;
    const double kernel_avg = kernel_sum / n;
    const double copy_avg = copy_sum / n;
    const double cleanup_avg = cleanup_sum / n;

    double prep_var = 0, kernel_var = 0, copy_var = 0, cleanup_var = 0;
    for (size_t i = 0; i < prep_times.size(); ++i) {
        prep_var += (prep_times[i] - prep_avg) * (prep_times[i] - prep_avg);
        kernel_var += (kernel_times[i] - kernel_avg) * (kernel_times[i] - kernel_avg);
        copy_var += (copy_times[i] - copy_avg) * (copy_times[i] - copy_avg);
        cleanup_var += (cleanup_times[i] - cleanup_avg) * (cleanup_times[i] - cleanup_avg);
    }

    const double prep_std = std::sqrt(prep_var / n);
    const double kernel_std = std::sqrt(kernel_var / n);
    const double copy_std = std::sqrt(copy_var / n);
    const double cleanup_std = std::sqrt(cleanup_var / n);

    printf("\nAverages over %zu repetitions:\n", prep_times.size());
    printf("  Avg preparation time: %.3f s (stddev: %.3f s)\n", prep_avg, prep_std);
    printf("  Avg kernel execution time: %.3f s (stddev: %.3f s)\n", kernel_avg, kernel_std);
    printf("  Avg data copy time: %.3f s (stddev: %.3f s)\n", copy_avg, copy_std);
    printf("  Avg memory deletion time: %.3f s (stddev: %.3f s)\n", cleanup_avg, cleanup_std);
    printf("  Whole time taken for %d reps: %.3f s\n", NUM_REPS, total_time);
    printf("=========================\n\n");
}

int main()
{
    performColdRun();

    std::vector<double> prep_times, kernel_times, copy_times, cleanup_times;
    prep_times.reserve(NUM_REPS);
    kernel_times.reserve(NUM_REPS);
    copy_times.reserve(NUM_REPS);
    cleanup_times.reserve(NUM_REPS);

    auto start_reps = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < NUM_REPS; ++rep) {
        runSingleTest(prep_times, kernel_times, copy_times, cleanup_times);
    }
    auto end_reps = std::chrono::high_resolution_clock::now();

    const double total_time = std::chrono::duration<double>(end_reps - start_reps).count();
    
    printResults(prep_times, kernel_times, copy_times, cleanup_times, total_time);
    
    cleanupMemory();
    return 0;
}