#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Constant memory for coefficients and parameters
__constant__ float d_coefficients[1024];
__constant__ float d_params[5];  // [tmin, tmax, length, coefficients, delta]

// Initialize constant memory
void initConstantMemory(int coefficients, float tmin, float tmax, int length, float delta) {
    if (coefficients > 1024) {
        std::cerr << "Error: coefficients > 1024 (" << coefficients << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    float h_coefficients[1024];
    for(int k = 1; k <= coefficients; ++k) {
        h_coefficients[k-1] = 1.0f / (4.0f * k * k - 4.0f * k + 1.0f);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_coefficients, h_coefficients, coefficients * sizeof(float)));

    float h_params[5] = {tmin, tmax, (float)length, (float)coefficients, delta};
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, h_params, 5 * sizeof(float)));
}

__global__ void fourier(int start_idx, int end_idx, float *results)
{
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end_idx) return;
    if (idx < 0) return;
    float t = d_params[0] + idx * d_params[4];  // tmin + idx * delta
    float sum = 0.0f;
    const float pi = 3.14159265f;
    int coeff = (int)d_params[3];
    if (coeff > 1024) coeff = 1024; // Defensive
    // Shared memory for intermediate calculations
    extern __shared__ float s_angles[];
    // Defensive: only threads < coeff fill shared memory
    for(int k = 1; k <= coeff; ++k) {
        s_angles[k-1] = (2 * k - 1) * pi * t;
    }
    __syncthreads();
    for(int k = 1; k <= coeff; ++k) {
        sum += cosf(s_angles[k-1]) * d_coefficients[k-1];
    }
    results[idx - start_idx] = 0.5f - (4.0f * sum) / (pi * pi);
}

int main()
{
    auto NUM_REPS = 5;
    for(auto rep = 0; rep < NUM_REPS; rep++) {

        const float tmin = -3.0f;
        const float tmax = 3.0f;
        
        //const int length = 1000000;
        // const int length = 10000000;
        // const int length = 100000000;
        const int length = 200000000;
        // const int length = 500000000;
        // const int length = 1000000000;
        // const int length = 2000000000;
        
        const int coefficients = 1024;
        const float delta = (tmax - tmin) / (length - 1);
        const int threadsPerBlock = 256;

        // Use a reasonable chunk size that fits in GPU memory
        const int chunkSize = 10000000; // 10 million

        // Initialize constant memory
        initConstantMemory(coefficients, tmin, tmax, length, delta);

        // Allocate device and host memory for one chunk
        float *d_results;
        CUDA_CHECK(cudaMalloc(&d_results, chunkSize * sizeof(float)));

        float *h_results;
        CUDA_CHECK(cudaHostAlloc(&h_results, chunkSize * sizeof(float), 
                    cudaHostAllocDefault | cudaHostAllocMapped));

        // Calculate shared memory size
        size_t sharedMemSize = coefficients * sizeof(float);

        // Process data in chunks
        for (int chunkStart = 0; chunkStart < length; chunkStart += chunkSize) {
            int chunkEnd = std::min(chunkStart + chunkSize, length);
            int thisChunkSize = chunkEnd - chunkStart;
            int blocks = (thisChunkSize + threadsPerBlock - 1) / threadsPerBlock;

            // Launch kernel for this chunk
            fourier<<<blocks, threadsPerBlock, sharedMemSize>>>(chunkStart, chunkEnd, d_results);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy results back to host
            CUDA_CHECK(cudaMemcpy(h_results, d_results, thisChunkSize * sizeof(float), cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaFree(d_results));
        CUDA_CHECK(cudaFreeHost(h_results));
    }

    return 0;
}