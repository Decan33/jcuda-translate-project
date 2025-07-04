#include <iostream>
// #include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <chrono>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_REPS = 5;

__global__ void computeKernel(
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
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    auto t = tmin + idx * delta;
    auto sum = 0.0f;
    for (auto k = 1; k <= coefficients; ++k) {
        auto angle = (2 * k - 1) * pi_over_T * t;
        auto denominator = 4.0f * k * k - 4.0f * k + 1.0f;
        sum += cosf(angle) / denominator;
    }
    results[idx] = T * 0.5f - result_coefficient * sum;
}

void performColdRun(float tmin, float delta, int length, int coefficients, 
                   float pi, float pi_over_T, float result_coefficient, float T) {
    float *d_results;
    cudaMalloc(&d_results, length * sizeof(float));

    auto threadsPerBlock = THREADS_PER_BLOCK;
    auto blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

    // Execute kernel without timing
    computeKernel<<<blocksPerGrid, threadsPerBlock>>>(
        tmin, delta, length, coefficients, pi, pi_over_T, result_coefficient, T, d_results
    );
    cudaDeviceSynchronize();

    // Copy results without timing
    float *h_results = new float[length];
    cudaMemcpy(h_results, d_results, length * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] h_results;
    cudaFree(d_results);
}

int main() {
    constexpr float tmin = -3.0f;
    constexpr float tmax = 3.0f;
    constexpr int length = 2000000000;
    constexpr int coefficients = 1024;
    constexpr float T = 1.0f;
    constexpr float delta = (tmax - tmin) / (length - 1);
    constexpr float pi = 3.14159265f;
    constexpr float pi_sq = pi * pi;
    constexpr float pi_over_T = pi / T;
    constexpr float result_coefficient = (4.0f * T) / pi_sq;

    // Cold run to warm up GPU
    printf("Performing cold run to warm up GPU...\n");
    performColdRun(tmin, delta, length, coefficients, pi, pi_over_T, result_coefficient, T);
    printf("Cold run completed.\n\n");

    std::vector<double> prep_times, kernel_times, delete_times;
    
    for (auto rep = 0; rep < NUM_REPS; ++rep) {
        auto prep_start = std::chrono::high_resolution_clock::now();
        
        float *d_results;
        cudaMalloc(&d_results, length * sizeof(float));

        auto threadsPerBlock = THREADS_PER_BLOCK;
        auto blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;
        
        auto prep_end = std::chrono::high_resolution_clock::now();
        prep_times.push_back(std::chrono::duration<double>(prep_end - prep_start).count());

        cudaEvent_t kernel_start, kernel_stop;
        cudaEventCreate(&kernel_start);
        cudaEventCreate(&kernel_stop);
        cudaEventRecord(kernel_start);

        computeKernel<<<blocksPerGrid, threadsPerBlock>>>(
            tmin, delta, length, coefficients, pi, pi_over_T, result_coefficient, T, d_results
        );
        cudaDeviceSynchronize();
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);
        
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);

        kernel_times.push_back(kernel_ms / 1000.0);

        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);

        auto delete_start = std::chrono::high_resolution_clock::now();
        
        float *h_results = new float[length];
        cudaMemcpy(h_results, d_results, length * sizeof(float), cudaMemcpyDeviceToHost);

        delete[] h_results;
        cudaFree(d_results);
        
        auto delete_end = std::chrono::high_resolution_clock::now();

        delete_times.push_back(std::chrono::duration<double>(delete_end - delete_start).count());
    }
    
    double prep_sum = 0, kernel_sum = 0, delete_sum = 0;
    
    for (auto i = 0u; i < prep_times.size(); ++i) {

        printf("Repetition %u:\n", i + 1);
        printf("  Preparation time: %.6f s\n", prep_times[i]);
        printf("  Kernel execution time: %.6f s\n", kernel_times[i]);
        printf("  Memory deletion time: %.6f s\n", delete_times[i]);
        prep_sum += prep_times[i];
        kernel_sum += kernel_times[i];
        delete_sum += delete_times[i];
    }

    auto n = static_cast<double>(prep_times.size());
    double prep_avg = prep_sum / n;
    double kernel_avg = kernel_sum / n;
    double delete_avg = delete_sum / n;
    double prep_var = 0, kernel_var = 0, delete_var = 0;
    
    for (auto i = 0u; i < prep_times.size(); ++i) {
        prep_var += (prep_times[i] - prep_avg) * (prep_times[i] - prep_avg);
        kernel_var += (kernel_times[i] - kernel_avg) * (kernel_times[i] - kernel_avg);
        delete_var += (delete_times[i] - delete_avg) * (delete_times[i] - delete_avg);
    }

    double prep_std = std::sqrt(prep_var / n);
    double kernel_std = std::sqrt(kernel_var / n);
    double delete_std = std::sqrt(delete_var / n);
    
    printf("\nAverages over %zu repetitions:\n", prep_times.size());
    printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prep_avg, prep_std);
    printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernel_avg, kernel_std);
    printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delete_avg, delete_std);
    printf("=========================\n\n");
    return 0;
}
