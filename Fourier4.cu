#include <cuda_runtime.h>
#include <cmath>
#include "cuda_c_files/data.h"

__constant__ float d_coefficients[MAX_COEFFICIENTS];
__constant__ float d_params[5];  // [tmin, tmax, length, coefficients, delta]

extern "C"
__global__ void fourier(int start_idx, int end_idx, float *results)
{
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end_idx) return;
    float t = d_params[0] + idx * d_params[4];
    float sum = 0.0f;
    constexpr float pi = 3.14159265f;
    int coeff = static_cast<int>(d_params[3]);
    if (coeff > MAX_COEFFICIENTS) coeff = MAX_COEFFICIENTS;
    extern __shared__ float s_angles[];
    for (auto k = 1; k <= coeff; ++k) {
        s_angles[k - 1] = (2 * k - 1) * pi * t;
    }
    __syncthreads();
    for (auto k = 1; k <= coeff; ++k) {
        sum += cosf(s_angles[k - 1]) * d_coefficients[k - 1];
    }
    results[idx - start_idx] = 0.5f - (4.0f * sum) / (pi * pi);
}
