extern "C"
{
    __global__ void dot(float *a, float *b, float *c) {
        __shared__ float cache[256]; // Use the actual number for threadsPerBlock
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int cacheIndex = threadIdx.x;

        float temp = 0.0;
        while (tid < 33 * 1024) { // Use the actual number for N
            temp += a[tid] * b[tid];
            tid += blockDim.x * gridDim.x;
        }

        cache[cacheIndex] = temp;
        __syncthreads();

        int i = blockDim.x / 2;
        while (i != 0) {
            if (cacheIndex < i)
                cache[cacheIndex] += cache[cacheIndex + i];
            __syncthreads();
            i /= 2;
        }

        if (cacheIndex == 0)
            c[blockIdx.x] = cache[0];
    }
}
