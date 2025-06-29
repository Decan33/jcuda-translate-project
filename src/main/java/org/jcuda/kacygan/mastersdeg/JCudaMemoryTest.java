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
        var hostData = new byte[(int) SIZE];
        var hPointer = Pointer.to(hostData);
        var deviceData = new Pointer();
        cudaMalloc(deviceData, SIZE);
        cudaMemset(deviceData, 0, SIZE);

        var start = new cudaEvent_t();
        var stop = new cudaEvent_t();
        cudaEventCreate(start);
        cudaEventCreate(stop);

        var startCPU = System.nanoTime();
        cudaEventRecord(start, null);

        cudaMemcpy(deviceData, hPointer, SIZE, cudaMemcpyHostToDevice);

        for (int i = 0; i < N; i++) {
            launchAddOneKernel(deviceData);
        }

        cudaMemcpy(hPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop, null);
        cudaEventSynchronize(stop);
        var elapsedTime = new float[1];
        cudaEventElapsedTime(elapsedTime, start, stop);
        var stopCPU = System.nanoTime();

        System.out.printf("Version 1 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
        System.out.printf("Version 1 GPU event time: %.3f s\n", elapsedTime[0] / 1000.0f);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(deviceData);
    }

    static void version2() {
        byte[] hostData = new byte[(int) SIZE];
        var hostPointer = Pointer.to(hostData);
        var deviceData = new Pointer();
        cudaMalloc(deviceData, SIZE);
        cudaMemset(deviceData, 0, SIZE);

        var start = new cudaEvent_t();
        var stop = new cudaEvent_t();
        cudaEventCreate(start);
        cudaEventCreate(stop);

        var startCPU = System.nanoTime();
        cudaEventRecord(start, null);

        for (int i = 0; i < N; i++) {
            cudaMemcpy(deviceData, hostPointer, SIZE, cudaMemcpyHostToDevice);
            launchAddOneKernel(deviceData);
            cudaMemcpy(hostPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(stop, null);
        cudaEventSynchronize(stop);
        var elapsedTime = new float[1];
        cudaEventElapsedTime(elapsedTime, start, stop);
        var stopCPU = System.nanoTime();

        System.out.printf("Version 2 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
        System.out.printf("Version 2 GPU event time: %.3f s\n", elapsedTime[0] / 1000.0f);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(deviceData);
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


