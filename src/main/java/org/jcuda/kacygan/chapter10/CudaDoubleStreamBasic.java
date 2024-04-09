package org.jcuda.kacygan.chapter10;

import jcuda.*;
import jcuda.driver.*;
import org.jcuda.kacygan.constants.CudaConstants;

import static jcuda.driver.JCudaDriver.*;


public class CudaDoubleStreamBasic {

    private CUmodule module;
    private CUfunction function;

    public CudaDoubleStreamBasic(String ptxFileName) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var device = new CUdevice();
        var context = new CUcontext();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        module = new CUmodule();
        function = new CUfunction();
        cuModuleLoad(module, ptxFileName);
        cuModuleGetFunction(function, module, "kernel");
    }

    public void performComputation(int[] hostA, int[] hostB, int[] hostC) {
        var devA = new CUdeviceptr();
        var devB = new CUdeviceptr();
        var devC = new CUdeviceptr();

        cuMemAlloc(devA, (long) CudaConstants.CHAPTER_10_N.size * Sizeof.INT);
        cuMemAlloc(devB, (long) CudaConstants.CHAPTER_10_N.size * Sizeof.INT);
        cuMemAlloc(devC, (long) CudaConstants.CHAPTER_10_N.size * Sizeof.INT);

        cuMemcpyHtoDAsync(devA, Pointer.to(hostA), (long) CudaConstants.CHAPTER_10_N.size * Sizeof.INT, new CUstream());
        cuMemcpyHtoDAsync(devB, Pointer.to(hostB), (long) CudaConstants.CHAPTER_10_N.size * Sizeof.INT, new CUstream());

        var blockSize = 256; // Example block size
        var numBlocks = CudaConstants.CHAPTER_10_N.size * Sizeof.INT / blockSize;

        cuLaunchKernel(function,
                numBlocks, 1, 1, // Grid dimension
                blockSize, 1, 1, // Block dimension
                0, new CUstream(), // Shared memory and stream
                null, null); // Kernel parameters and extra

        cuMemcpyDtoHAsync(Pointer.to(hostC), devC, (long) CudaConstants.CHAPTER_10_N.size * Sizeof.INT, new CUstream());

    }
}
