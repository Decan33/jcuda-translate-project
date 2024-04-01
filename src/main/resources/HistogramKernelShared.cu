extern "C"
{
    __global__ void histoKernel(unsigned char *buffer, long size, unsigned int *histo) {
        __shared__ unsigned int temp[256];
        temp[threadIdx.x] = 0;
        __syncthreads();

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        while (i < size) {
            atomicAdd(&temp[buffer[i]], 1);
            i += stride;
        }

        __syncthreads();
        atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
    }
}
