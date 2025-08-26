#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
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

__constant__ float d_coefficients[MAX_COEFFICIENTS];
__constant__ float d_params[5];  // [tmin, tmax, length, coefficients, delta]

void initConstantMemory(int coefficients, float tmin, float tmax, int length, float delta) {
    if (coefficients > MAX_COEFFICIENTS) {
        std::cerr << "Error: coefficients > " << MAX_COEFFICIENTS << " (" << coefficients << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    float h_coefficients[MAX_COEFFICIENTS];
    for (int k = 1; k <= coefficients; ++k) {
        h_coefficients[k - 1] = 1.0f / (4.0f * k * k - 4.0f * k + 1.0f);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_coefficients, h_coefficients, coefficients * sizeof(float)));
    float h_params[5] = {tmin, tmax, static_cast<float>(length), static_cast<float>(coefficients), delta};
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, h_params, 5 * sizeof(float)));
}

__global__ void fourier(int start_idx, int end_idx, float *results)
{
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end_idx) return;
    float t = d_params[0] + idx * d_params[4];
    float sum = 0.0f;
    extern __shared__ float s_angles[];
    int coeff = static_cast<int>(d_params[3]);
    if (coeff > MAX_COEFFICIENTS) coeff = MAX_COEFFICIENTS;
    constexpr float pi = 3.14159265f;
    for (int k = 1; k <= coeff; ++k) {
        s_angles[k - 1] = (2 * k - 1) * pi * t;
    }
    __syncthreads();
    for (int k = 1; k <= coeff; ++k) {
        sum += cosf(s_angles[k - 1]) * d_coefficients[k - 1];
    }
    results[idx - start_idx] = 0.5f - (4.0f * sum) / (pi * pi);
}

void performColdRun(float tmin, float tmax, int length, int coefficients, float delta) {
    initConstantMemory(coefficients, tmin, tmax, length, delta);
    int chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    float* d_results[NUM_STREAMS];
    float* h_results[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_results[i], chunkSize * sizeof(float)));
        CUDA_CHECK(cudaHostAlloc(&h_results[i], chunkSize * sizeof(float), cudaHostAllocDefault));
    }
    auto sharedMemSize = coefficients * sizeof(float);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int startIdx = i * chunkSize;
        int endIdx = std::min(startIdx + chunkSize, length);
        int thisChunk = endIdx - startIdx;
        int blocks = (thisChunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fourier<<<blocks, THREADS_PER_BLOCK, sharedMemSize, streams[i]>>>(startIdx, endIdx, d_results[i]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_results[i], d_results[i], thisChunk * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaFreeHost(h_results[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

int main()
{
    printf("Performing cold run to warm up GPU...\n");
    performColdRun(tmin, tmax, length, coefficients, delta);
    printf("Cold run completed.\n\n");
    std::vector<double> prep_times, kernel_times, copy_times, delete_times;
    int chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;
    auto start_reps = std::chrono::high_resolution_clock::now();
    for (auto rep = 0; rep < NUM_REPS; rep++) {
        auto prep_start = std::chrono::high_resolution_clock::now();
        initConstantMemory(coefficients, tmin, tmax, length, delta);
        cudaStream_t streams[NUM_STREAMS];
        float* d_results[NUM_STREAMS];
        float* h_results[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&d_results[i], chunkSize * sizeof(float)));
            CUDA_CHECK(cudaHostAlloc(&h_results[i], chunkSize * sizeof(float), cudaHostAllocDefault));
        }
        auto sharedMemSize = coefficients * sizeof(float);
        auto prep_end = std::chrono::high_resolution_clock::now();
        prep_times.push_back(std::chrono::duration<double>(prep_end - prep_start).count());
        cudaEvent_t kernel_start, kernel_stop;
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));
        CUDA_CHECK(cudaEventRecord(kernel_start));
        for (int i = 0; i < NUM_STREAMS; ++i) {
            int startIdx = i * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, length);
            int thisChunk = endIdx - startIdx;
            int blocks = (thisChunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            fourier<<<blocks, THREADS_PER_BLOCK, sharedMemSize, streams[i]>>>(startIdx, endIdx, d_results[i]);
            CUDA_CHECK(cudaGetLastError());
        }
        cudaEvent_t copy_start, copy_stop;
        CUDA_CHECK(cudaEventCreate(&copy_start));
        CUDA_CHECK(cudaEventCreate(&copy_stop));
        CUDA_CHECK(cudaEventRecord(copy_start));
        for (int i = 0; i < NUM_STREAMS; ++i) {
            int startIdx = i * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, length);
            int thisChunk = endIdx - startIdx;
            CUDA_CHECK(cudaMemcpyAsync(h_results[i], d_results[i], thisChunk * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        }
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        CUDA_CHECK(cudaEventRecord(copy_stop));
        CUDA_CHECK(cudaEventSynchronize(copy_stop));
        float copy_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&copy_ms, copy_start, copy_stop));
        copy_times.push_back(copy_ms / 1000.0);
        CUDA_CHECK(cudaEventDestroy(copy_start));
        CUDA_CHECK(cudaEventDestroy(copy_stop));
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms / 1000.0);
        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
        auto delete_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaFree(d_results[i]));
            CUDA_CHECK(cudaFreeHost(h_results[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        auto delete_end = std::chrono::high_resolution_clock::now();
        delete_times.push_back(std::chrono::duration<double>(delete_end - delete_start).count());
    }
    auto end_reps = std::chrono::high_resolution_clock::now();
    double prep_sum = 0, kernel_sum = 0, delete_sum = 0, copy_sum = 0;
    printf("\n===== Timing Summary =====\n");
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
        delete_sum += delete_times[i];
        copy_sum += copy_times[i];
    }
    printf("\nAverages over %zu repetitions:\n", prep_times.size());
    printf("  Avg preparation time: %.6f s\n", prep_sum / prep_times.size());
    printf("  Avg kernel execution time: %.6f s\n", kernel_sum / kernel_times.size());
    printf("  Avg data copy time: %.6f s\n", copy_sum / copy_times.size());
    printf("  Avg memory deletion time: %.6f s\n", delete_sum / delete_times.size());
    printf("  Whole time taken for %d reps: %.6f s\n", NUM_REPS, std::chrono::duration<double>(end_reps - start_reps).count());
    printf("=========================\n\n");
    return 0;
}