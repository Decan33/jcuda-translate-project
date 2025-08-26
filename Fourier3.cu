#include <cuda_runtime.h>
#include <cmath>

extern "C"
__global__ void fourier(
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
    if (idx_local >= stream_size) return;
    int idx_global = idx_local + stream_offset;
    float t = tmin + idx_global * delta;
    float sum = 0.0f;
    for (int k = 1; k <= coefficients; ++k) {
        float angle = (2 * k - 1) * pi_over_T * t;
        float denominator = 4.0f * k * k - 4.0f * k + 1.0f;
        sum += cosf(angle) / denominator;
    }
    results[idx_local] = T * 0.5f - result_coefficient * sum;
}
