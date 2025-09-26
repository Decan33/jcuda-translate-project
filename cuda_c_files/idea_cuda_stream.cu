#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <algorithm>
#include "data.h"

__global__ void fourier(
    float tmin,
    float delta,
    int length,
    int coefficients,
    float pi,
    float pi_over_T,
    float result_coefficient,
    float T,
    float *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    float t = tmin + idx * delta;
    float sum = 0.0f;
    for (int k = 1; k <= coefficients; ++k) {
        float angle = (2 * k - 1) * pi_over_T * t;
        float denominator = 4.0f * k * k - 4.0f * k + 1.0f;
        sum += cosf(angle) / denominator;
    }
    results[idx] = T * 0.5f - result_coefficient * sum;
}

static inline int grid_for(int n, int block) {
    return (n + block - 1) / block;
}


static cudaStream_t streams[NUM_STREAMS];
static float* d_results[NUM_STREAMS];
static float* h_results[NUM_STREAMS];
static bool memory_initialized = false;
static int chunkSize;

void initializeMemory() {
    if (memory_initialized) return;

    chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&d_results[i], (size_t)chunkSize * sizeof(float)));
        CUDA_CHECK(cudaHostAlloc(&h_results[i], (size_t)chunkSize * sizeof(float), cudaHostAllocDefault));
    }
    memory_initialized = true;
}

void performColdRun() {
    initializeMemory();

    for (int i = 0; i < NUM_STREAMS; ++i) {
        const int startIdx         = i * chunkSize;
        const int currentChunkSize = std::min(chunkSize, length - startIdx);
        if (currentChunkSize <= 0) break;
        const int blocksPerGrid    = grid_for(currentChunkSize, THREADS_PER_BLOCK);

        fourier<<<blocksPerGrid, THREADS_PER_BLOCK, 0, streams[i]>>>(
            tmin + startIdx * delta, delta, currentChunkSize, coefficients, pi, pi_over_T, result_coefficient, T, d_results[i]);
        CUDA_CHECK(cudaGetLastError());

    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
}

void runTest() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        const int startIdx         = i * chunkSize;
        const int currentChunkSize = std::min(chunkSize, length - startIdx);
        if (currentChunkSize <= 0) break;
        const int blocksPerGrid    = grid_for(currentChunkSize, THREADS_PER_BLOCK);

        fourier<<<blocksPerGrid, THREADS_PER_BLOCK, 0, streams[i]>>>(
            tmin + startIdx * delta, delta, currentChunkSize, coefficients, pi, pi_over_T, result_coefficient, T, d_results[i]);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            h_results[i], d_results[i],
            (size_t)currentChunkSize * sizeof(float),
            cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
}

int main() {
    printf("Performing cold run to warm up GPU...\n");
    performColdRun();
    printf("Cold run completed.\n\n");

    auto start_reps = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < NUM_REPS; rep++) {
        runTest();
    }
    auto end_reps = std::chrono::high_resolution_clock::now();

    printf("\n===== Timing Summary =====\n");
    printf("=========================\n\n");
    printf("  Whole time taken for %d reps: %.6f s\n",
           NUM_REPS, std::chrono::duration<double>(end_reps - start_reps).count());
    printf("=========================\n\n");

    // FIX 6: Cleanup at the very end
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaFreeHost(h_results[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return 0;
}