package org.jcuda.kacygan.mastersdeg.example;

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
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

public class SimpleDeviceCall {
    private static final String ADD_KERNEL_PTX = "AddKernel.ptx";
    private static final String KERNEL_FUNCTION_NAME = "add";
    public static void main(String[] args) {
        cuInit(0);
        JCudaDriver.setExceptionsEnabled(true);

        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ADD_KERNEL_PTX);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, KERNEL_FUNCTION_NAME);

        var a = 2;
        var b = 7;
        var resultFromDevice = new int[1];
        CUdeviceptr deviceC = new CUdeviceptr();
        cuMemAlloc(deviceC, Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{a}),
                Pointer.to(new int[]{b}),
                Pointer.to(deviceC)
        );

        cuLaunchKernel(function,
                1, 1, 1,
                1, 1, 1,
                0, null,
                kernelParameters, null
        );

        cuMemcpyDtoH(Pointer.to(resultFromDevice), deviceC, Sizeof.INT);
        System.out.printf("2 + 7 = %d%n", resultFromDevice[0]);

        cuMemFree(deviceC);
        cuCtxDestroy(context);
    }
}

