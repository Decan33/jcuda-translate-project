package org.jcuda.kacygan.chapter5;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

public class DotProduct {
    private static final int N = 33 * 1024;
    public static final int SIZE = N * Sizeof.FLOAT;
    private static final int THREADS_PER_BLOCK = 256;
    private static final int BLOCKS_PER_GRID = Math.min(32, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    public static final int BLOCKS_PER_GRID_SIZE = BLOCKS_PER_GRID * Sizeof.FLOAT;

    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Load the kernel
        var module = new CUmodule();
        cuModuleLoad(module, "DotKernel.ptx");
        var function = new CUfunction();
        cuModuleGetFunction(function, module, "dot");

        // Allocate memory on the CPU
        var hostA = new float[N];
        var hostB = new float[N];
        var hostPartialC = new float[BLOCKS_PER_GRID];

        // Initialize host arrays
        for (var i = 0; i < N; i++) {
            hostA[i] = i;
            hostB[i] = i * 2;
        }

        // Allocate and copy memory to the GPU
        var devA = new CUdeviceptr();
        var devB = new CUdeviceptr();
        var devPartialC = new CUdeviceptr();
        cuMemAlloc(devA, SIZE);
        cuMemAlloc(devB, SIZE);
        cuMemAlloc(devPartialC, BLOCKS_PER_GRID_SIZE);
        cuMemcpyHtoD(devA, Pointer.to(hostA), SIZE);
        cuMemcpyHtoD(devB, Pointer.to(hostB), SIZE);

        // Execute the kernel
        Pointer kernelParameters = Pointer.to(Pointer.to(devA), Pointer.to(devB), Pointer.to(devPartialC));
        cuLaunchKernel(function,
                BLOCKS_PER_GRID, 1, 1, // Grid dimension
                THREADS_PER_BLOCK, 1, 1, // Block dimension
                0, null, // Shared memory size and stream
                kernelParameters, null); // Kernel- and extra parameters
        cuCtxSynchronize();

        // Copy the result back to the host
        cuMemcpyDtoH(Pointer.to(hostPartialC), devPartialC, BLOCKS_PER_GRID_SIZE);

        // Finish up on the CPU side
        var c = 0f;
        for (var i = 0; i < BLOCKS_PER_GRID; i++) {
            c += hostPartialC[i];
        }

        // Output the result
        System.out.printf("Does GPU value %.6g = %.6g?\n", c, 2 * sumSquares(N - 1));

        // Clean up
        cuMemFree(devA);
        cuMemFree(devB);
        cuMemFree(devPartialC);
        cuCtxDestroy(context);
    }

    private static float sumSquares(float x) {
        return x * (x + 1) * (2 * x + 1) / 6;
    }
}
