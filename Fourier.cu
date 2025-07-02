extern "C"
__global__ void fourier(float tmin, float delta, int length, int coefficients, float *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    float t = tmin + idx * delta;
    float sum = 0.0f;
    const float pi = 3.14159265f;
    for (int k = 1; k <= coefficients; ++k)
    {
        float angle = (2 * k - 1) * pi * t;
        sum += cosf(angle) / (4.0f * k * k - 4.0f * k + 1.0f);
    }
    float pi_sq = pi * pi;
    results[idx] = 0.5f - (4.0f * sum) / pi_sq;
}