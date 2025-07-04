#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <chrono>
// #include <fstream>

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_COEFFICIENTS = 1024;
constexpr int NUM_REPS = 5;

__constant__ float const_tmin;
__constant__ float const_delta;
__constant__ int   const_coefficients;

__constant__ float const_pi;
__constant__ float const_pi_squared;
__constant__ float const_T;
__constant__ float const_pi_over_T;
__constant__ float constant_result_coefficient;

__global__ void computeKernel(float* results)
{
    __shared__ float shared_coefficients[MAX_COEFFICIENTS];

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    auto idx = threadIdx.x;

    for (auto k = idx; k < const_coefficients; k += blockDim.x)
    {
        auto denominator = 4.0f * (k + 1) * (k + 1) - 4.0f * (k + 1) + 1.0f;
        shared_coefficients[k] = 1.0f / denominator;
    }

    __syncthreads();

    if (tid >= gridDim.x * blockDim.x) return;

    auto t = const_tmin + tid * const_delta;
    auto sum = 0.0f;

    for (auto k = 1; k <= const_coefficients; ++k)
    {
        auto angle = (2 * k - 1) * const_pi_over_T * t;
        auto numerator = cosf(angle);
        sum += numerator * shared_coefficients[k - 1];
    }

    results[tid] = const_T * 0.5f - (constant_result_coefficient * sum);
}

void performColdRun(float tmin, float tmax, int length, int coefficients, float delta) {
    float* result_device;
    cudaMalloc((void**)&result_device, length * sizeof(float));

    auto blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    constexpr float pi = 3.14159265f;
    constexpr float pi_squared = pi * pi;
    constexpr float T = 1.0f;
    constexpr float pi_over_T = pi / T;
    constexpr float host_result_coefficient = (4.0f * T) / pi_squared;

    cudaMemcpyToSymbol(const_tmin,   &tmin, sizeof(float));
    cudaMemcpyToSymbol(const_delta,  &delta, sizeof(float));
    cudaMemcpyToSymbol(const_coefficients, &coefficients, sizeof(int));

    cudaMemcpyToSymbol(const_pi, &pi, sizeof(float));
    cudaMemcpyToSymbol(const_pi_squared, &pi_squared, sizeof(float));
    cudaMemcpyToSymbol(const_T, &T, sizeof(float));
    cudaMemcpyToSymbol(const_pi_over_T, &pi_over_T, sizeof(float));
    cudaMemcpyToSymbol(constant_result_coefficient, &host_result_coefficient, sizeof(float));

    // Execute kernel without timing
    computeKernel<<<blocks, THREADS_PER_BLOCK>>>(result_device);
    cudaDeviceSynchronize();

    // Copy results without timing
    float* result_host = new float[length];
    cudaMemcpy(result_host, result_device, length * sizeof(float), cudaMemcpyDeviceToHost);
    delete[] result_host;
    cudaFree(result_device);
}

int main()
{
	constexpr float tmin = -3.0f;
	constexpr float tmax = 3.0f;
	constexpr int length = 200000000;
	//const int length = 500000000;
	//const int length = 1000000000;
	//const int length = 2000000000;
	
	constexpr int coefficients = 1024;

	constexpr float delta = (tmax - tmin) / (length - 1);

	// Cold run to warm up GPU
	printf("Performing cold run to warm up GPU...\n");
	performColdRun(tmin, tmax, length, coefficients, delta);
	printf("Cold run completed.\n\n");

	std::vector<double> prep_times, kernel_times, delete_times;
	
	for (auto rep = 0; rep < NUM_REPS; ++rep) {
		auto prep_start = std::chrono::high_resolution_clock::now();
		
		float* result_device;
		cudaMalloc((void**)&result_device, length * sizeof(float));

		auto blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		constexpr float pi = 3.14159265f;
		constexpr float pi_squared = pi * pi;
		constexpr float T = 1.0f;
		constexpr float pi_over_T = pi / T;
		constexpr float host_result_coefficient = (4.0f * T) / pi_squared;

		cudaMemcpyToSymbol(const_tmin,   &tmin, sizeof(float));
		cudaMemcpyToSymbol(const_delta,  &delta, sizeof(float));
		cudaMemcpyToSymbol(const_coefficients, &coefficients, sizeof(int));

		cudaMemcpyToSymbol(const_pi, &pi, sizeof(float));
		cudaMemcpyToSymbol(const_pi_squared, &pi_squared, sizeof(float));
		cudaMemcpyToSymbol(const_T, &T, sizeof(float));
		cudaMemcpyToSymbol(const_pi_over_T, &pi_over_T, sizeof(float));
		cudaMemcpyToSymbol(constant_result_coefficient, &host_result_coefficient, sizeof(float));

		auto prep_end = std::chrono::high_resolution_clock::now();
		prep_times.push_back(std::chrono::duration<double>(prep_end - prep_start).count());
		
		cudaEvent_t kernel_start, kernel_stop;
		cudaEventCreate(&kernel_start);
		cudaEventCreate(&kernel_stop);
		cudaEventRecord(kernel_start);
		
		computeKernel<<<blocks, THREADS_PER_BLOCK>>>(result_device);
		cudaDeviceSynchronize();
		cudaEventRecord(kernel_stop);
		cudaEventSynchronize(kernel_stop);
		
		float kernel_ms = 0.0f;
		cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop);
		kernel_times.push_back(kernel_ms / 1000.0);
		
		cudaEventDestroy(kernel_start);
		cudaEventDestroy(kernel_stop);
		
		auto delete_start = std::chrono::high_resolution_clock::now();
		
		float* result_host = new float[length];
		cudaMemcpy(result_host, result_device, length * sizeof(float), cudaMemcpyDeviceToHost);
		delete[] result_host;
		cudaFree(result_device);
		
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