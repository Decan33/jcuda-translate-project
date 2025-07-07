#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <locale>
#include "data.h"

enum HostMemoryMode {
    PINNED,
    PAGEABLE_MALLOC,
    PAGEABLE_VECTOR
};

constexpr int MIN_EXPONENT = 10;
constexpr int MAX_EXPONENT = 28;
constexpr int HOST_ALLOC_FLAGS = cudaHostAllocWriteCombined;

struct CopyResult {
    float elapsedTimeMs;
    float bandwidthMBps;
};

CopyResult computeCopyTimeAndBandwidth(HostMemoryMode mode, int memorySize) {
    void *hostData = nullptr;
    void *deviceData = nullptr;
    cudaError_t err;
    float elapsedTimeMs = 0.0f;
    float totalBytes = 0.0f;
    float bandwidthMBps = 0.0f;
    
    std::vector<char> vec;
    switch (mode) {
        case PINNED:
            err = cudaHostAlloc(&hostData, memorySize, HOST_ALLOC_FLAGS);
            if (err != cudaSuccess) {
                std::cerr << "cudaHostAlloc failed: " << cudaGetErrorString(err) << std::endl;
                return {0.0f, 0.0f};
            }
            break;
        case PAGEABLE_MALLOC:
            hostData = malloc(memorySize);
            if (!hostData) {
                std::cerr << "malloc failed" << std::endl;
                return {0.0f, 0.0f};
            }
            break;
        case PAGEABLE_VECTOR:
            vec.resize(memorySize);
            hostData = vec.data();
            break;
    }
    
    char *data = static_cast<char*>(hostData);
    for (int i = 0; i < memorySize; i++) {
        data[i] = static_cast<char>(i);
    }
    
    err = cudaMalloc(&deviceData, memorySize);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < NUM_REPS; i++) {
        err = cudaMemcpyAsync(deviceData, hostData, memorySize, cudaMemcpyHostToDevice, nullptr);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
            goto cleanup;
        }
    }
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&elapsedTimeMs, start, stop);
    totalBytes = static_cast<float>(memorySize) * NUM_REPS;
    bandwidthMBps = (totalBytes / (elapsedTimeMs / 1000.0f)) / (1024.0f * 1024.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
cleanup:
    if (deviceData) {
        cudaFree(deviceData);
    }
    
    if (hostData) {
        if (mode == PINNED) {
            cudaFreeHost(hostData);
        } else if (mode == PAGEABLE_MALLOC) {
            free(hostData);
        }
    }
    
    return {elapsedTimeMs, bandwidthMBps};
}

void runTest(HostMemoryMode mode) {
    std::cout << "Running bandwidth test for ";
    switch (mode) {
        case PINNED:
            std::cout << "PINNED memory";
            break;
        case PAGEABLE_MALLOC:
            std::cout << "PAGEABLE_MALLOC memory";
            break;
        case PAGEABLE_VECTOR:
            std::cout << "PAGEABLE_VECTOR memory";
            break;
    }
    std::cout << std::endl;
    
    std::cout << std::setw(12) << "Size (bytes)" << " | "
              << std::setw(15) << "Time (s)" << " | "
              << std::setw(15) << "Bandwidth (MB/s)" << std::endl;
    std::cout << std::string(48, '-') << std::endl;
    std::vector<int> memorySizes;
    std::vector<CopyResult> results;
    
    for (int i = 0; i < MAX_EXPONENT - MIN_EXPONENT; i++) {
        int memorySize = 1 << (MIN_EXPONENT + i);
        memorySizes.push_back(memorySize);
        
        CopyResult result = computeCopyTimeAndBandwidth(mode, memorySize);
        results.push_back(result);
    }
    
    for (size_t i = 0; i < memorySizes.size(); i++) {
        std::cout << std::setw(12) << memorySizes[i] << " | "
                  << std::setw(15) << std::fixed << std::setprecision(6) << (results[i].elapsedTimeMs / 1000.0f) << " | "
                  << std::setw(15) << std::fixed << std::setprecision(3) << results[i].bandwidthMBps << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    cudaSetDevice(0);
    
    runTest(PINNED);
    runTest(PAGEABLE_MALLOC);
    runTest(PAGEABLE_VECTOR);
    
    std::cout << "Done" << std::endl;
    return 0;
} 