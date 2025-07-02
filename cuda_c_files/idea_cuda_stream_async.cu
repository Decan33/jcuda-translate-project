#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>

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
    cudaMemcpyToSymbol(d_coefficients, h_coefficients, coefficients * sizeof(float));

    float h_params[5] = {tmin, tmax, (float)length, (float)coefficients, delta};
    cudaMemcpyToSymbol(d_params, h_params, 5 * sizeof(float));
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
    if (coeff > 1024) coeff = 1024;

    extern __shared__ float s_angles[];

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

        initConstantMemory(coefficients, tmin, tmax, length, delta);

        float *d_results;
        cudaMalloc(&d_results, chunkSize * sizeof(float));

        float *h_results;
        cudaHostAlloc(&h_results, chunkSize * sizeof(float),
                    cudaHostAllocDefault | cudaHostAllocMapped);

        size_t sharedMemSize = coefficients * sizeof(float);

        for (int chunkStart = 0; chunkStart < length; chunkStart += chunkSize) {
            int chunkEnd = std::min(chunkStart + chunkSize, length);
            int thisChunkSize = chunkEnd - chunkStart;
            int blocks = (thisChunkSize + threadsPerBlock - 1) / threadsPerBlock;

            fourier<<<blocks, threadsPerBlock, sharedMemSize>>>(chunkStart, chunkEnd, d_results);
            cudaDeviceSynchronize();

            cudaMemcpy(h_results, d_results, thisChunkSize * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(d_results);
        cudaFreeHost(h_results);
    }

    return 0;
}