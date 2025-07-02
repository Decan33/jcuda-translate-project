#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define MAX_COEFFICIENTS 1024
#define NUM_STREAMS 4
#define BATCH_SIZE (16 * 1024 * 1024)


__global__ void computeKernel(
    float tmin,
    float delta,
    int length,
    int coefficients,
    float pi,
    float pi_over_T,
    float result_coefficient,
    float T,
    float *results,
    int stream_offset,
    int stream_size)
{
    int idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_local >= stream_size)
        return;

    int idx_global = idx_local + stream_offset;
    float t = tmin + idx_global * delta;
    float sum = 0.0f;

    for (int k = 1; k <= coefficients; ++k)
    {
        float angle = (2 * k - 1) * pi_over_T * t;
        float denominator = 4.0f * k * k - 4.0f * k + 1.0f;
        sum += cosf(angle) / denominator;
    }

    results[idx_local] = T * 0.5f - result_coefficient * sum;
}

int main()
{
	auto NUM_REPS = 5;
	for(auto rep = 0; rep < NUM_REPS; rep++) {
		const float tmin = -3.0f;
		const float tmax = 3.0f;
		
		const int length = 200000000;
		// const int length = 500000000;
		// const int length = 1000000000;
		// const int length = 2000000000;
		
		const int coefficients = 1024;
		const float T = 1.0f;
		const float delta = (tmax - tmin) / (length - 1);

		const float pi = 3.14159265f;
		const float pi_sq = pi * pi;
		const float pi_over_T = pi / T;
		const float result_coefficient = (4.0f * T) / pi_sq;

		cudaStream_t streams[NUM_STREAMS];
		for (int i = 0; i < NUM_STREAMS; i++) {
			cudaStreamCreate(&streams[i]);
		}

		float *d_results[NUM_STREAMS][2];
		float *h_results[NUM_STREAMS][2];
		for (int i = 0; i < NUM_STREAMS; i++) {
			for (int b = 0; b < 2; b++) {
				cudaMalloc((void**)&d_results[i][b], BATCH_SIZE * sizeof(float));
				cudaMallocHost((void**)&h_results[i][b], BATCH_SIZE * sizeof(float));
			}
		}
		
		float *final_results = new float[length];
		
		int threadsPerBlock = THREADS_PER_BLOCK;
		int num_batches = (length + (BATCH_SIZE * NUM_STREAMS) - 1) / (BATCH_SIZE * NUM_STREAMS);
		for (int batch = 0; batch < num_batches; ++batch) {
			for (int s = 0; s < NUM_STREAMS; ++s) {
				int global_offset = (batch * NUM_STREAMS + s) * BATCH_SIZE;
				if (global_offset >= length) continue;
				int current_batch_size = (global_offset + BATCH_SIZE > length) ? (length - global_offset) : BATCH_SIZE;
				int buf = batch % 2;
				int blocksPerGrid = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;
				computeKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
					tmin,
					delta,
					length,
					coefficients,
					pi,
					pi_over_T,
					result_coefficient,
					T,
					d_results[s][buf],
					global_offset,
					current_batch_size
				);
				cudaMemcpyAsync(
					h_results[s][buf],
					d_results[s][buf],
					current_batch_size * sizeof(float),
					cudaMemcpyDeviceToHost,
					streams[s]
				);
			}
			
			if (batch > 0) {
				int prev_buf = (batch - 1) % 2;
				for (int s = 0; s < NUM_STREAMS; ++s) {
					int prev_offset = ((batch - 1) * NUM_STREAMS + s) * BATCH_SIZE;
					if (prev_offset >= length) continue;
					int prev_batch_size = (prev_offset + BATCH_SIZE > length) ? (length - prev_offset) : BATCH_SIZE;
					cudaStreamSynchronize(streams[s]);
					std::memcpy(final_results + prev_offset, h_results[s][prev_buf], prev_batch_size * sizeof(float));
				}
			}
		}
		
		int last_batch = num_batches - 1;
		int last_buf = last_batch % 2;
		for (int s = 0; s < NUM_STREAMS; ++s) {
			int last_offset = (last_batch * NUM_STREAMS + s) * BATCH_SIZE;
			if (last_offset >= length) continue;
			int last_batch_size = (last_offset + BATCH_SIZE > length) ? (length - last_offset) : BATCH_SIZE;
			cudaStreamSynchronize(streams[s]);
			std::memcpy(final_results + last_offset, h_results[s][last_buf], last_batch_size * sizeof(float));
		}

		for (int i = 0; i < NUM_STREAMS; i++) {
			for (int b = 0; b < 2; b++) {
				cudaFree(d_results[i][b]);
				cudaFreeHost(h_results[i][b]);
			}
		}
		delete[] final_results;
		
		for (int i = 0; i < NUM_STREAMS; i++) {
			cudaStreamDestroy(streams[i]);
		}
	}

    return 0;
}
