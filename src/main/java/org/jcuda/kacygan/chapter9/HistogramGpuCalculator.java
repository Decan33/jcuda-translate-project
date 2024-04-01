package org.jcuda.kacygan.chapter9;

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
import static jcuda.driver.JCudaDriver.cuMemsetD8;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

public class HistogramGpuCalculator {
    private static final String PTX_FILE_NAME = "histoKernel.ptx";

    public static int[] calculateHistogram(byte[] data) {
        // Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Load the kernel
        var module = new CUmodule();
        cuModuleLoad(module, PTX_FILE_NAME);
        var function = new CUfunction();
        cuModuleGetFunction(function, module, "histoKernel");

        // Allocate and copy memory to the GPU
        var deviceBuffer = new CUdeviceptr();
        cuMemAlloc(deviceBuffer, data.length);
        cuMemcpyHtoD(deviceBuffer, Pointer.to(data), data.length);

        var histogram = new int[256];
        var deviceHistogram = new CUdeviceptr();
        cuMemAlloc(deviceHistogram, histogram.length * Sizeof.INT);
        cuMemsetD8(deviceHistogram, (byte) 0, histogram.length * Sizeof.INT);

        // Kernel launch parameters
        int blocks = 256; // Example value; adjust based on your needs
        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceBuffer),
                Pointer.to(new long[]{data.length}),
                Pointer.to(deviceHistogram)
        );

        cuLaunchKernel(function,
                blocks, 1, 1, // Grid dimension
                256, 1, 1,    // Block dimension
                0, null,      // Shared memory size and stream
                kernelParameters, null); // Kernel- and extra parameters
        cuCtxSynchronize();

        // Copy the result back to host
        cuMemcpyDtoH(Pointer.to(histogram), deviceHistogram, histogram.length * Sizeof.INT);

        // Cleanup
        cuMemFree(deviceBuffer);
        cuMemFree(deviceHistogram);
        cuCtxDestroy(context);

        return histogram;
    }
}
