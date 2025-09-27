package org.jcuda.kacygan.mastersdeg;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public class JCudaMemoryTest implements FourierTest {

    static final long SIZE = 1L * 1024 * 1024 * 1024;
    public static final int BLOCKS = (int) ((SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    private double kernelSum = 0.0;
    private double hostToDeviceTransferSum = 0.0;
    private double deviceToHostTransferSum = 0.0;

    MemoryType memoryTypeUsed;
    private static CUmodule module = new CUmodule();
    private static CUfunction function = new CUfunction();

    public JCudaMemoryTest(MemoryType memoryType) {
        memoryTypeUsed = memoryType;
    }

    @Override
    public void runTest() {
        initJCuda();

        System.out.println("Testing " + memoryTypeUsed + " memory mode");

        for (int i = 0; i < 5; i++) {
            version1();
            version2();
        }
    }

    private void initJCuda() {
        JCuda.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        var context = new CUcontext();
        cuCtxCreate(context, 0, device);

        cuModuleLoad(module, "AddOneKernel.ptx");
        cuModuleGetFunction(function, module, "addOneKernel");
    }

    private void version1() {
        var hPtr = new Pointer();
        var dPtr = new Pointer();

        if (MemoryType.NORMAL.equals(memoryTypeUsed)) {
            var hostData = new byte[(int) SIZE];
            Arrays.fill(hostData, (byte) 0);

            hPtr = Pointer.to(hostData);
        } else {
            //We use pinned memory
            cudaHostAlloc(hPtr, SIZE, cudaHostAllocDefault);
            cudaMemset(hPtr, 0, SIZE);
        }

        cudaMalloc(dPtr, SIZE);

        var start = System.nanoTime();

        doFirstVersionCopies(hPtr, dPtr);

        var end = System.nanoTime();

        cudaFree(dPtr);
        if (MemoryType.PINNED.equals(memoryTypeUsed)) {
            cudaFreeHost(hPtr);
        }


        System.out.printf("[Kernel] sum: %.3f s, avg: %.3f s\n", kernelSum, kernelSum / REPS);
        System.out.printf("[H->D] sum: %.3f s, avg %.3f s\n", hostToDeviceTransferSum, hostToDeviceTransferSum / REPS);
        System.out.printf("[D->H] sum: %.3f s, avg %.3f s\n", deviceToHostTransferSum, deviceToHostTransferSum / REPS);
        System.out.printf("Version 1 CPU time: %.3f s\n", toSeconds(end, start));
        clearTimings();
    }

    private void doFirstVersionCopies(Pointer hPointer, Pointer deviceData) {
        for (int rep = 0; rep < REPS; rep++) {

            var firstTransferStart = System.nanoTime();
            cudaMemcpy(deviceData, hPointer, SIZE, cudaMemcpyHostToDevice);
            var firstTransferEnd = System.nanoTime();


            var kernelStart = System.nanoTime();
            for (int i = 0; i < N; i++) {
                launchAddOneKernel(deviceData);
            }
            var kernelEnd = System.nanoTime();

            var secondTransferStart = System.nanoTime();
            cudaMemcpy(hPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);
            var secondTransferEnd = System.nanoTime();

            kernelSum += toSeconds(kernelEnd, kernelStart);
            hostToDeviceTransferSum += toSeconds(firstTransferEnd, firstTransferStart);
            deviceToHostTransferSum += toSeconds(secondTransferEnd, secondTransferStart);
        }
    }

    private void version2() {
        var hPtr = new Pointer();
        var dPtr = new Pointer();

        if (MemoryType.NORMAL.equals(memoryTypeUsed)) {
            var hostData = new byte[(int) SIZE];
            Arrays.fill(hostData, (byte) 0);

            hPtr = Pointer.to(hostData);
        } else {
            //We use pinned memory
            cudaHostAlloc(hPtr, SIZE, cudaHostAllocDefault);
            cudaMemset(hPtr, 0, SIZE);
        }

        cudaMalloc(dPtr, SIZE);

        var startCPU = System.nanoTime();

        doSecondVersionCopies(hPtr, dPtr);

        var stopCPU = System.nanoTime();

        cudaFree(dPtr);
        if (MemoryType.PINNED.equals(memoryTypeUsed)) {
            cudaFreeHost(hPtr);
        }

        System.out.printf("[Kernel] sum: %.3f s, avg: %.3f s\n", kernelSum, kernelSum / REPS);
        System.out.printf("[H->D] sum: %.3f s, avg %.3f s\n", hostToDeviceTransferSum, hostToDeviceTransferSum / REPS * N);
        System.out.printf("[D->H] sum: %.3f s, avg %.3f s\n", deviceToHostTransferSum, deviceToHostTransferSum / REPS * N);
        System.out.printf("Version 2 CPU time: %.3f s\n", toSeconds(stopCPU, startCPU));
        clearTimings();
    }

    private void doSecondVersionCopies(Pointer hostPointer, Pointer deviceData) {
        for (int rep = 0; rep < REPS; rep++) {

            var kernelStart = System.nanoTime();
            for (int i = 0; i < N; i++) {
                var firstTransferStart = System.nanoTime();
                cudaMemcpy(deviceData, hostPointer, SIZE, cudaMemcpyHostToDevice);
                var firstTransferEnd = System.nanoTime();

                launchAddOneKernel(deviceData);

                var secondTransferStart = System.nanoTime();
                cudaMemcpy(hostPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);
                var secondTransferEnd = System.nanoTime();

                hostToDeviceTransferSum += toSeconds(firstTransferEnd, firstTransferStart);
                deviceToHostTransferSum += toSeconds(secondTransferEnd, secondTransferStart);
            }
            var kernelEnd = System.nanoTime();

            kernelSum += toSeconds(kernelEnd, kernelStart);
        }
    }

    void launchAddOneKernel(Pointer data) {
        var kernelParams = Pointer.to(
                Pointer.to(data),
                Pointer.to(new long[]{SIZE})
        );

        cuLaunchKernel(function,
                BLOCKS, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                0, null,
                kernelParams, null);

        cuCtxSynchronize();
    }

    void clearTimings() {
        kernelSum = 0.0;
        hostToDeviceTransferSum = 0.0;
        deviceToHostTransferSum = 0.0;
    }

    double toSeconds(long end, long start) {
        return (end - start) / 1e9;
    }
}


