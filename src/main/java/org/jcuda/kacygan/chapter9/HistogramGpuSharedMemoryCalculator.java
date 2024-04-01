package org.jcuda.kacygan.chapter9;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaDeviceProp;

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
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

public class HistogramGpuSharedMemoryCalculator {
    private static final int SIZE = 100 * 1024 * 1024;
    private static final int NUM_BINS = 256;
    private static final String PTX_FILE = "HistogramKernel.ptx";

    public static void calculateHistogram(byte[] data) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        var module = new CUmodule();
        cuModuleLoad(module, PTX_FILE);
        var function = new CUfunction();
        cuModuleGetFunction(function, module, "histoKernel");

        var deviceData = new CUdeviceptr();
        cuMemAlloc(deviceData, SIZE);
        cuMemcpyHtoD(deviceData, Pointer.to(data), SIZE);

        var deviceHistogram = new CUdeviceptr();
        cuMemAlloc(deviceHistogram, NUM_BINS * Sizeof.INT);
        cuMemsetD8(deviceHistogram, (byte) 0, NUM_BINS * Sizeof.INT);

        var prop = new cudaDeviceProp();
        cudaGetDeviceProperties(prop, 0);
        var blocks = prop.multiProcessorCount * 2;

        cuLaunchKernel(function,
                blocks, 1, 1,
                256, 1, 1,
                0, null,
                Pointer.to(deviceData, Pointer.to(new long[]{SIZE}), deviceHistogram), null);
        cuCtxSynchronize();

        var histogram = new int[NUM_BINS];
        cuMemcpyDtoH(Pointer.to(histogram), deviceHistogram, NUM_BINS * Sizeof.INT);

        cuMemFree(deviceData);
        cuMemFree(deviceHistogram);
        cuCtxDestroy(context);
    }
}
