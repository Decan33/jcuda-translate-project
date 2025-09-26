#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
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

static void setAllConstants()
{
    CUDA_CHECK(cudaMemcpyToSymbol(const_delta, &delta, sizeof(delta)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_coefficients, &coefficients, sizeof(coefficients)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi, &pi, sizeof(pi)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi_squared, &pi_squared, sizeof(pi_squared)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_T, &T, sizeof(T)));
    CUDA_CHECK(cudaMemcpyToSymbol(const_pi_over_T, &pi_over_T, sizeof(pi_over_T)));
    CUDA_CHECK(cudaMemcpyToSymbol(constant_result_coefficient, &result_coefficient, sizeof(result_coefficient)));
}

static cudaStream_t streams[NUM_STREAMS];
static float* dOut[NUM_STREAMS];
static float* hStage[NUM_STREAMS];
static bool memory_initialized = false;
static int chunkSize;

void initializeMemory() {
    if (memory_initialized) return;

    chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;
    const size_t chunkBytes = (size_t)chunkSize * sizeof(float);

    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&dOut[s], chunkBytes));
        CUDA_CHECK(cudaHostAlloc(&hStage[s], chunkBytes, cudaHostAllocDefault));
    }
    memory_initialized = true;
}

void performColdRun()
{
    initializeMemory();
    setAllConstants();
    CUDA_CHECK(cudaMemcpyToSymbol(const_tmin, &tmin, sizeof(tmin)));

    for (int s = 0; s < NUM_STREAMS; ++s) {
        const int offset = s * chunkSize;
        const int currentChunk = (offset + chunkSize <= length)
                                 ? chunkSize : (length - offset);

        float chunk_tmin = tmin + offset * delta;
        CUDA_CHECK(cudaMemcpyToSymbol(const_tmin, &chunk_tmin, sizeof(chunk_tmin)));

        const dim3 block(THREADS_PER_BLOCK);
        const dim3 grid((currentChunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        fourier<<<grid, block, 0, streams[s]>>>(dOut[s]);
        CUDA_CHECK(cudaGetLastError());

    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

}

void runTest()
{

    for (int s = 0; s < NUM_STREAMS; ++s) {
        const int offset = s * chunkSize;
        const int currentChunk = (offset + chunkSize <= length)
                                 ? chunkSize : (length - offset);
        const size_t bytes = (size_t)currentChunk * sizeof(float);

        float chunk_tmin = tmin + offset * delta;
        CUDA_CHECK(cudaMemcpyToSymbol(const_tmin, &chunk_tmin, sizeof(chunk_tmin)));

        const dim3 block(THREADS_PER_BLOCK);
        const dim3 grid((currentChunk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        fourier<<<grid, block, 0, streams[s]>>>(dOut[s]);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            hStage[s], dOut[s], bytes, cudaMemcpyDeviceToHost, streams[s]));
    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

}

int main()
{
    std::puts("Performing cold run to warm up GPU...");
    performColdRun();
    std::puts("Cold run completed.\n");

    auto start_reps = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < NUM_REPS; rep++) {
        runTest();
    }
    auto end_reps = std::chrono::high_resolution_clock::now();

    std::puts("\n===== Timing Summary =====");
    std::printf("Whole time taken for %d reps: %.3f s\n",
                NUM_REPS,
                std::chrono::duration<double>(end_reps - start_reps).count());
    std::puts("=========================\n");

    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaFree(dOut[s]));
        CUDA_CHECK(cudaFreeHost(hStage[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    return 0;
}