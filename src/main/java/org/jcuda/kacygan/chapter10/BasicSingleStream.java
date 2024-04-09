package org.jcuda.kacygan.chapter10;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
import static jcuda.runtime.JCuda.cudaFreeHost;
import static jcuda.runtime.JCuda.cudaHostAlloc;
import static jcuda.runtime.JCuda.cudaHostAllocDefault;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToHost;

public class BasicSingleStream {
    private static final int N = 1024 * 1024;
    private static final int FULL_DATA_SIZE = N * 20;
    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        CUcontext context = new CUcontext();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Allocate memory on the GPU
        CUdeviceptr devA = new CUdeviceptr();
        CUdeviceptr devB = new CUdeviceptr();
        CUdeviceptr devC = new CUdeviceptr();
        cuMemAlloc(devA, N * Sizeof.INT);
        cuMemAlloc(devB, N * Sizeof.INT);
        cuMemAlloc(devC, N * Sizeof.INT);

        // Allocate host memory (pinned memory for efficiency)
        Pointer hostA = new Pointer();
        Pointer hostB = new Pointer();
        Pointer hostC = new Pointer();
        cudaHostAlloc(hostA, FULL_DATA_SIZE * Sizeof.INT, cudaHostAllocDefault);
        cudaHostAlloc(hostB, FULL_DATA_SIZE * Sizeof.INT, cudaHostAllocDefault);
        cudaHostAlloc(hostC, FULL_DATA_SIZE * Sizeof.INT, cudaHostAllocDefault);

        int[] hostAArray = new int[FULL_DATA_SIZE];
        int[] hostBArray = new int[FULL_DATA_SIZE];
        int[] hostCArray = new int[FULL_DATA_SIZE];
        // Fill hostAArray and hostBArray with random data
        for (int i = 0; i < FULL_DATA_SIZE; i++) {
            hostAArray[i] = (int) (Math.random() * Integer.MAX_VALUE);
            hostBArray[i] = (int) (Math.random() * Integer.MAX_VALUE);
        }
        cudaMemcpy(Pointer.to(hostAArray), hostA, FULL_DATA_SIZE * Sizeof.INT, cudaMemcpyHostToHost);
        cudaMemcpy(Pointer.to(hostBArray), hostB, FULL_DATA_SIZE * Sizeof.INT, cudaMemcpyHostToHost);

        // Loading the module and kernel function
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "kernel.ptx");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "kernel");
        int blockSize = 256; // This can be adjusted depending on the kernel requirements and GPU architecture
        int gridSize = (N + blockSize - 1) / blockSize;

        // Event and stream creation for timing and asynchronous execution
        CUevent start = new CUevent();
        CUevent stop = new CUevent();
        cuEventCreate(start, 0);
        cuEventCreate(stop, 0);
        CUstream stream = new CUstream();
        cuStreamCreate(stream, 0);

        cuEventRecord(start, stream);
        for (int i = 0; i < FULL_DATA_SIZE; i += N) {
            // Copy segments of host memory to device
            cuMemcpyHtoDAsync(devA, Pointer.to(Arrays.copyOfRange(hostAArray, i, i + N)), N * Sizeof.INT, stream);
            cuMemcpyHtoDAsync(devB, Pointer.to(Arrays.copyOfRange(hostBArray, i, i + N)), N * Sizeof.INT, stream);

            // Set up kernel parameters
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(devA),
                    Pointer.to(devB),
                    Pointer.to(devC)
            );

            // Launch the kernel
            cuLaunchKernel(function,
                    gridSize, 1, 1,      // Grid dimension
                    blockSize, 1, 1,   // Block dimension
                    0, stream,          // Shared memory size and stream
                    kernelParameters, null // Kernel parameters and extra
            );

            // Copy from device back to host
            cuMemcpyDtoHAsync(Pointer.to(Arrays.copyOfRange(hostCArray, i, i + N)), devC, N * Sizeof.INT, stream);
        }
        cuStreamSynchronize(stream);
        cuEventRecord(stop, stream);
        cuEventSynchronize(stop);

        // Cleanup
        cuMemFree(devA);
        cuMemFree(devB);
        cuMemFree(devC);
        cudaFreeHost(hostA);
        cudaFreeHost(hostB);
        cudaFreeHost(hostC);
        cuStreamDestroy(stream);
        cuCtxDestroy(context);
    }

}
