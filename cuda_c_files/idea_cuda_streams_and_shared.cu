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

__constant__ float const_coeffs[MAX_COEFFICIENTS];

__global__ void fourier(float* results)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= gridDim.x * blockDim.x) return;

    float t = const_tmin + tid * const_delta;
    float sum = 0.0f;

    for (auto k = 1; k <= const_coefficients; ++k)
    {
        float angle = (2 * k - 1) * const_pi_over_T * t;
        float numerator = cosf(angle);

        sum += numerator * const_coeffs[k - 1];
    }

    results[tid] = const_T * 0.5f - (constant_result_coefficient * sum);
}

#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

static inline dim3 blocksFor(size_t n, int tpb) {
    return dim3(static_cast<unsigned>((n + tpb - 1) / tpb));
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

    {
        std::vector<float> hcoeff(MAX_COEFFICIENTS, 0.0f);
        for (int k = 0; k < coefficients; ++k) {
            float kp1 = float(k + 1);
            float denominator = 4.0f * kp1 * kp1 - 4.0f * kp1 + 1.0f;
            hcoeff[k] = 1.0f / denominator;
        }
        CUDA_CHECK(cudaMemcpyToSymbol(const_coeffs, hcoeff.data(), coefficients * sizeof(float)));
    }
}

void performColdRun()
{

    setAllConstants();
    CUDA_CHECK(cudaMemcpyToSymbol(const_tmin, &tmin, sizeof(tmin)));

    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, length * sizeof(float)));

    cudaStream_t sCompute, copyStreams;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copyStreams,    cudaStreamNonBlocking));

    float* hPinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&hPinned, length * sizeof(float), cudaHostAllocPortable));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid  = blocksFor(length, THREADS_PER_BLOCK);

    fourier<<<grid, block, 0, sCompute>>>(dOut);
    CUDA_CHECK(cudaMemcpyAsync(hPinned, dOut, length*sizeof(float), cudaMemcpyDeviceToHost, copyStreams));

    CUDA_CHECK(cudaStreamSynchronize(sCompute));
    CUDA_CHECK(cudaStreamSynchronize(copyStreams));

    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFreeHost(hPinned));
    CUDA_CHECK(cudaStreamDestroy(sCompute));
    CUDA_CHECK(cudaStreamDestroy(copyStreams));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runTest()
{
    setAllConstants();

    std::vector<float> hResult(length);

    cudaStream_t sCompute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));

    cudaStream_t copyStreams[NUM_STREAMS];
    float* deviceBuffer[NUM_STREAMS] = {nullptr};
    float* hStage[NUM_STREAMS] = {nullptr};
    size_t nCount[NUM_STREAMS] = {0};
    size_t startIdx[NUM_STREAMS] = {0};

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStreams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&deviceBuffer[i], CHUNK_BYTES));
        CUDA_CHECK(cudaHostAlloc(&hStage[i], CHUNK_BYTES, cudaHostAllocDefault));
    }

    size_t base = 0;
    size_t chunkIdx = 0;

    while (base < length) {
        const size_t s = chunkIdx % NUM_STREAMS;

        if (chunkIdx >= NUM_STREAMS) {
            size_t bytes = nCount[s] * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(hStage[s], deviceBuffer[s], bytes, cudaMemcpyDeviceToHost, copyStreams[s]));
            CUDA_CHECK(cudaStreamSynchronize(copyStreams[s]));
            std::memcpy(&hResult[startIdx[s]], hStage[s], bytes);
        }

        const size_t nThis = std::min<size_t>(CHUNK_ELEMS, length - base);
        nCount[s] = nThis;
        startIdx[s] = base;

        const float tminChunk = tmin + base * delta;
        CUDA_CHECK(cudaMemcpyToSymbolAsync(const_tmin, &tminChunk, sizeof(tminChunk), 0, cudaMemcpyHostToDevice, sCompute));

        dim3 block(THREADS_PER_BLOCK);
        dim3 grid  = blocksFor(nThis, THREADS_PER_BLOCK);

        fourier<<<grid, block, 0, sCompute>>>(deviceBuffer[s]);
        CUDA_CHECK(cudaGetLastError());

        base += nThis;
        ++chunkIdx;
    }

    int totalChunks = (length + CHUNK_ELEMS - 1) / CHUNK_ELEMS;
    int startFlush  = (totalChunks > NUM_STREAMS) ? (totalChunks - NUM_STREAMS) : 0;

    for (int i = startFlush; i < totalChunks; ++i) {
        const size_t s = i % NUM_STREAMS;
    
        size_t bytes = nCount[s] * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(hStage[s], deviceBuffer[s], bytes, cudaMemcpyDeviceToHost, copyStreams[s]));
        CUDA_CHECK(cudaStreamSynchronize(copyStreams[s]));
        std::memcpy(&hResult[startIdx[s]], hStage[s], bytes);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaFree(deviceBuffer[i]));
        CUDA_CHECK(cudaFreeHost(hStage[i]));
        CUDA_CHECK(cudaStreamDestroy(copyStreams[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(sCompute));
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
    return 0;
}
