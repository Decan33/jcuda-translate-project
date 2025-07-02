#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define MAX_COEFFICIENTS 1024

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

    for (int k = 1; k <= const_coefficients; ++k)
    {
        float angle = (2 * k - 1) * const_pi_over_T * t;
        float numerator = cosf(angle);
        sum += numerator * shared_coefficients[k - 1];
    }

    results[tid] = const_T * 0.5f - (constant_result_coefficient * sum);
}

int main()
{
	auto NUM_REPS = 5;
	for(auto rep = 0; rep < NUM_REPS; rep++) {
		float tmin = -3.0f;
		float tmax = 3.0f;
		

		const int length = 200000000;
		//const int length = 500000000;
		//const int length = 1000000000;
		//const int length = 2000000000;
		
		const int coefficients = 1024;

		float delta = (tmax - tmin) / (length - 1);

		float* result_device;
		cudaMalloc((void**)&result_device, length * sizeof(float));

		int blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		float pi = 3.14159265f;
		float pi_squared = pi * pi;
		float T = 1.0f;
		float pi_over_T = pi / T;
		float host_result_coefficient = (4.0f * T) / pi_squared;

		cudaMemcpyToSymbol(const_tmin,   &tmin,sizeof(float));
		cudaMemcpyToSymbol(const_delta,  &delta,     sizeof(float));
		cudaMemcpyToSymbol(const_coefficients, &coefficients, sizeof(int));

		cudaMemcpyToSymbol(const_pi, &pi, sizeof(float));
		cudaMemcpyToSymbol(const_pi_squared, &pi_squared, sizeof(float));
		cudaMemcpyToSymbol(const_T, &T, sizeof(float));
		cudaMemcpyToSymbol(const_pi_over_T, &pi_over_T, sizeof(float));
		cudaMemcpyToSymbol(constant_result_coefficient, &host_result_coefficient, sizeof(float));

		computeKernel<<<blocks, THREADS_PER_BLOCK>>>(result_device);

		float* result_host = new float[length];
		cudaMemcpy(result_host, result_device, length * sizeof(float), cudaMemcpyDeviceToHost);

		delete[] result_host;
		cudaFree(result_device);
	}

    return 0;
}