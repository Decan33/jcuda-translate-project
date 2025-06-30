#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>

// Constant memory for coefficients and parameters
__constant__ float d_coefficients[1024];
__constant__ float d_params[5];  // [tmin, tmax, length, coefficients, delta]

// Initialize constant memory
void initConstantMemory(int coefficients, float tmin, float tmax, int length, float delta) {
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

    float t = d_params[0] + idx * d_params[4];  // tmin + idx * delta
    float sum = 0.0f;
    const float pi = 3.14159265f;
    
    // Shared memory for intermediate calculations
    extern __shared__ float s_angles[];
    
    // Calculate angles in shared memory
    for(int k = 1; k <= (int)d_params[3]; ++k) {  // coefficients
        s_angles[k-1] = (2 * k - 1) * pi * t;
    }
    __syncthreads();
    
    // Use shared memory and constant memory for calculations
    for(int k = 1; k <= (int)d_params[3]; ++k) {  // coefficients
        sum += cosf(s_angles[k-1]) * d_coefficients[k-1];
    }
    
    results[idx] = 0.5f - (4.0f * sum) / (pi * pi);
}

int main()
{
    const float tmin = -3.0f;
    const float tmax = 3.0f;
	
    //const int length = 1000000;
    // const int length = 10000000;
    // const int length = 100000000;
	// const int length = 200000000;
    // const int length = 500000000;
	//const int length = 1000000000;
	const int length = 2000000000;
	
    const int coefficients = 1024;
    const float delta = (tmax - tmin) / (length - 1);
    const int threadsPerBlock = 256;

    // Get device properties for optimal chunking
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    const int numSMs = prop.multiProcessorCount;
    
    // Calculate optimal chunk size based on device capabilities
    // Aim to keep all SMs busy with enough blocks
    const int blocksPerSM = 32; // Theoretical occupancy target
    const int totalBlocks = numSMs * blocksPerSM;
    const int N = threadsPerBlock * totalBlocks;

    // Use more streams for better overlap
    const int numStreams = 8;
    cudaStream_t streams[8];
    for(int i = 0; i < numStreams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Initialize constant memory
    initConstantMemory(coefficients, tmin, tmax, length, delta);

    float *d_results;
    cudaMalloc(&d_results, length * sizeof(float));

    float *h_results;
    cudaHostAlloc(&h_results, length * sizeof(float), 
                cudaHostAllocDefault | cudaHostAllocMapped);

    // Calculate shared memory size
    size_t sharedMemSize = coefficients * sizeof(float);

    // Process data in chunks using multiple streams with better overlap
    const int streamChunkSize = N;
    const int batchSize = numStreams * streamChunkSize;
    
    for (int batchStart = 0; batchStart < length; batchStart += batchSize) {
        for (int s = 0; s < numStreams; s++) {
            int chunkStart = batchStart + s * streamChunkSize;
            if (chunkStart >= length) break;
            
            int chunkEnd = std::min(chunkStart + streamChunkSize, length);
            int chunkSize = chunkEnd - chunkStart;
            
            if (chunkSize > 0) {
                int blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
                
                // Prefetch next chunk's data
                if (chunkStart + batchSize < length) {
                    cudaMemPrefetchAsync(d_results + chunkStart + batchSize, 
                                       std::min(streamChunkSize, length - (chunkStart + batchSize)) * sizeof(float),
                                       0, streams[s]);
                }
                
                // Launch kernel
                fourier<<<blocks, threadsPerBlock, sharedMemSize, streams[s]>>>
                    (chunkStart, chunkEnd, d_results);
                
                // Asynchronous memory transfer
                cudaMemcpyAsync(&h_results[chunkStart], 
                              &d_results[chunkStart],
                              chunkSize * sizeof(float),
                              cudaMemcpyDeviceToHost, 
                              streams[s]);
            }
        }
        
        // Optional: Synchronize every few batches to prevent too much queuing
        if (batchStart % (batchSize * 4) == 0) {
            cudaDeviceSynchronize();
        }
    }

    cudaDeviceSynchronize();

    // std::ofstream file("results_1024coeff.csv");
    // file << "t,f\n";
    // file.precision(6);
    // file << std::fixed;
    // for (int i = 0; i < length; ++i) {
    //     float t = tmin + i * delta;
    //     file << t << "," << h_results[i] << "\n";
    // }
    // file.close();

    cudaFree(d_results);
    cudaFreeHost(h_results);
    for(int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}