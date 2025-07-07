package org.jcuda.kacygan.mastersdeg;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public class JCudaMemoryTest {

    static final long SIZE = 1L * 1024 * 1024 * 1024;
    static final int N = 10;

    public static void main(String[] args) {
        JCuda.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        var context = new CUcontext();
        cuCtxCreate(context, 0, device);

        version1();
        version2();
    }

    static void version1() {
        var startCPU = System.nanoTime();

        for (int rep = 0; rep < 10; rep++) {
            var hostData = new byte[(int) SIZE];
            var hPointer = Pointer.to(hostData);
            var deviceData = new Pointer();
            cudaMalloc(deviceData, SIZE);
            cudaMemset(deviceData, 0, SIZE);


            cudaMemcpy(deviceData, hPointer, SIZE, cudaMemcpyHostToDevice);

            for (int i = 0; i < N; i++) {
                launchAddOneKernel(deviceData);
            }

            cudaMemcpy(hPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);

        }

        var stopCPU = System.nanoTime();

        System.out.printf("Version 1 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
    }

    static void version2() {
        var startCPU = System.nanoTime();
        for (int rep = 0; rep < 10; rep++) {
            byte[] hostData = new byte[(int) SIZE];
            var hostPointer = Pointer.to(hostData);
            var deviceData = new Pointer();
            cudaMalloc(deviceData, SIZE);
            cudaMemset(deviceData, 0, SIZE);

            for (int i = 0; i < N; i++) {
                cudaMemcpy(deviceData, hostPointer, SIZE, cudaMemcpyHostToDevice);
                launchAddOneKernel(deviceData);
                cudaMemcpy(hostPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);
            }

            cudaFree(deviceData);
        }
        var stopCPU = System.nanoTime();


        System.out.printf("Version 2 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
    }

    static void launchAddOneKernel(Pointer data) {
        var threadsPerBlock = 256;
        var blocksPerGrid = (int)((JCudaMemoryTest.SIZE + threadsPerBlock - 1) / threadsPerBlock);

        var module = new CUmodule();
        cuModuleLoad(module, "AddOneKernel.ptx");

        var function = new CUfunction();
        cuModuleGetFunction(function, module, "addOneKernel");

        var kernelParams = Pointer.to(
                Pointer.to(data),
                Pointer.to(new long[]{JCudaMemoryTest.SIZE})
        );

        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                threadsPerBlock, 1, 1,
                0, null,
                kernelParams, null);

        cuCtxSynchronize();
    }
}


