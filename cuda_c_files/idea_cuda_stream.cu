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

    std::vector<double> prep_times, kernel_times, copy_times, delete_times;
    int chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;
    auto start_reps = std::chrono::high_resolution_clock::now();
    for (auto rep = 0; rep < NUM_REPS; ++rep) {
        auto prep_start = std::chrono::high_resolution_clock::now();
        cudaStream_t streams[NUM_STREAMS];
        float* d_results[NUM_STREAMS];
        float* h_results[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&d_results[i], chunkSize * sizeof(float)));
            CUDA_CHECK(cudaHostAlloc(&h_results[i], chunkSize * sizeof(float), cudaHostAllocDefault));
        }
        auto prep_end = std::chrono::high_resolution_clock::now();
        prep_times.push_back(std::chrono::duration<double>(prep_end - prep_start).count());
        cudaEvent_t kernel_start, kernel_stop;
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));
        CUDA_CHECK(cudaEventRecord(kernel_start));
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

        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms / 1000.0);

        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
        auto copy_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_STREAMS; ++i) {
            int startIdx = i * chunkSize;
            int currentChunkSize = std::min(chunkSize, length - startIdx);
            CUDA_CHECK(cudaMemcpy(h_results[i], d_results[i], currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost));
        }

        auto copy_end = std::chrono::high_resolution_clock::now();
        copy_times.push_back(std::chrono::duration<double>(copy_end - copy_start).count());
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
    double prep_sum = 0, kernel_sum = 0, copy_sum = 0, delete_sum = 0;
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

    printf("=========================\n\n");
    printf("\nAverages over %zu repetitions:\n", prep_times.size());
    printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prep_avg, prep_std);
    printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernel_avg, kernel_std);
    printf("  Avg data copy time: %.6f s (stddev: %.6f s)\n", copy_avg, copy_std);
    printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delete_avg, delete_std);
    printf("  Whole time taken for %d reps: %.6f s\n", NUM_REPS, std::chrono::duration<double>(end_reps - start_reps).count());
    printf("=========================\n\n");

    return 0;
}
