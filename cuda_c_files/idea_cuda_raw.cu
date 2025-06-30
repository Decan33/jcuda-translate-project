#include <iostream>
// #include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <string>

#define THREADS_PER_BLOCK 256
#define MAX_COEFFICIENTS 1024

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length)
        return;

    float t = tmin + idx * delta;
    float sum = 0.0f;

    for (int k = 1; k <= coefficients; ++k)
    {
        float angle = (2 * k - 1) * pi_over_T * t;
        float denominator = 4.0f * k * k - 4.0f * k + 1.0f;
        sum += cosf(angle) / denominator;
    }

    results[idx] = T * 0.5f - result_coefficient * sum;
}

int main()
{
	auto NUM_REPS = 5;
	for(int rep = 0; rep < NUM_REPS; rep++) {
	
		const float tmin = -3.0f;
		const float tmax = 3.0f;
		// const int length = 65536 * 16;

		//const int length = 200000000;
		//const int length = 500000000;
		//const int length = 1000000000;
		const int length = 2000000000;
		
		const int coefficients = 1024;
		const float T = 1.0f;
		const float delta = (tmax - tmin) / (length - 1);

		const float pi = 3.14159265f;
		const float pi_sq = pi * pi;
		const float pi_over_T = pi / T;
		const float result_coefficient = (4.0f * T) / pi_sq;

		float *d_results;
		cudaMalloc((void**)&d_results, length * sizeof(float));

		int threadsPerBlock = THREADS_PER_BLOCK;
		int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

		computeKernel<<<blocksPerGrid, threadsPerBlock>>>(
			tmin,
			delta,
			length,
			coefficients,
			pi,
			pi_over_T,
			result_coefficient,
			T,
			d_results
		);
		cudaDeviceSynchronize();

		float *h_results = new float[length];
		cudaMemcpy(h_results, d_results, length * sizeof(float), cudaMemcpyDeviceToHost);


		/*
		
			
		std::ofstream file("results_1024coeff.csv");
		file << "t,f\n";
		file.precision(6);
		file << std::fixed;
		for (int i = 0; i < length; ++i)
		{
			float t = tmin + i * delta;
			file << t << "," << h_results[i] << "\n";
		}
		file.close();
	*/
		cudaFree(d_results);
		delete[] h_results;

		// std::cout << "Computation complete. Results saved to results.csv.\n";
		
	}
    
    return 0;
}
