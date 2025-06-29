extern "C"
__global__ void addOneKernel(unsigned char* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1;
    }
}