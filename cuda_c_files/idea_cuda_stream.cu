#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
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

__global__ void fourier(
    float tmin,
    float delta,
    int length,
    int coefficients,
    float pi,
    float pi_over_T,
    float result_coefficient,
    float T,
    float *results,
    int stream_offset,
    int stream_size)
{
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_local >= stream_size) return;
    int idx_global = idx_local + stream_offset;
    float t = tmin + idx_global * delta;
    float sum = 0.0f;
    for (int k = 1; k <= coefficients; ++k) {
        float angle = (2 * k - 1) * pi_over_T * t;
        float denominator = 4.0f * k * k - 4.0f * k + 1.0f;
        sum += cosf(angle) / denominator;
    }
    results[idx_local] = T * 0.5f - result_coefficient * sum;
}

void performColdRun(float tmin, float tmax, int length, int coefficients, float T, 
                   float delta, float pi, float pi_over_T, float result_coefficient) {
    int chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    float* d_results[NUM_STREAMS];
    float* h_results[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMalloc(&d_results[i], chunkSize * sizeof(float)));
        CUDA_CHECK(cudaHostAlloc(&h_results[i], chunkSize * sizeof(float), cudaHostAllocDefault));
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int startIdx = i * chunkSize;
        int currentChunkSize = std::min(chunkSize, length - startIdx);
        int blocksPerGrid = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fourier<<<blocksPerGrid, THREADS_PER_BLOCK, 0, streams[i]>>>(
            tmin, delta, length, coefficients, pi, pi_over_T, result_coefficient, T,
            d_results[i], startIdx, currentChunkSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int startIdx = i * chunkSize;
        int currentChunkSize = std::min(chunkSize, length - startIdx);
        CUDA_CHECK(cudaMemcpy(h_results[i], d_results[i], currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost));
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(d_results[i]));
        CUDA_CHECK(cudaFreeHost(h_results[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

int main() {
    printf("Performing cold run to warm up GPU...\n");
    performColdRun(tmin, tmax, length, coefficients, T, delta, pi, pi_over_T, result_coefficient);
    printf("Cold run completed.\n\n");
    std::vector<float> hAll(length);

    int chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;
    auto start_reps = std::chrono::high_resolution_clock::now();
    for (auto rep = 0; rep < NUM_REPS; ++rep) {
        cudaStream_t streams[NUM_STREAMS];
        float* d_results[NUM_STREAMS];
        float* h_results[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&d_results[i], chunkSize * sizeof(float)));
            CUDA_CHECK(cudaHostAlloc(&h_results[i], chunkSize * sizeof(float), cudaHostAllocDefault));
        }

        for (int i = 0; i < NUM_STREAMS; ++i) {
            int startIdx = i * chunkSize;
            int currentChunkSize = std::min(chunkSize, length - startIdx);
            int blocksPerGrid = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            fourier<<<blocksPerGrid, THREADS_PER_BLOCK, 0, streams[i]>>>(
                tmin, delta, length, coefficients, pi, pi_over_T, result_coefficient, T,
                d_results[i], startIdx, currentChunkSize
            );
            CUDA_CHECK(cudaGetLastError());
        }
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        for (int i = 0; i < NUM_STREAMS; ++i) {
            int startIdx = i * chunkSize;
            int currentChunkSize = std::min(chunkSize, length - startIdx);
            CUDA_CHECK(cudaMemcpy(h_results[i], d_results[i], currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost));
        }

        for (int i = 0; i < NUM_STREAMS; ++i) {
            const int startIdx  = i * chunkSize;
            const int nThis     = std::min(chunkSize, length - startIdx);
            if (nThis <= 0) break;

            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

            std::memcpy(hAll.data() + startIdx, h_results[i],
                        (size_t)nThis * sizeof(float));
        }

        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaFree(d_results[i]));
            CUDA_CHECK(cudaFreeHost(h_results[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
    }

    auto end_reps = std::chrono::high_resolution_clock::now();
    printf("\n===== Timing Summary =====\n");


    printf("=========================\n\n");
    printf("  Whole time taken for %d reps: %.6f s\n", NUM_REPS, std::chrono::duration<double>(end_reps - start_reps).count());
    printf("=========================\n\n");

    return 0;
}
