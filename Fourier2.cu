#define MAX_COEFFICIENTS 1024

__constant__ float const_tmin;
__constant__ float const_delta;
__constant__ int   const_coefficients;
__constant__ float const_pi;
__constant__ float const_pi_squared;
__constant__ float const_T;
__constant__ float const_pi_over_T;
__constant__ float constant_result_coefficient;
__constant__ int const_chunk_size;

extern "C"
__global__ void fourier(float* results)
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

    if (tid >= const_chunk_size) return;

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