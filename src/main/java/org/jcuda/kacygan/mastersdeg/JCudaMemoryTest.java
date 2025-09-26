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
    static final Integer THREADS_PER_BLOCK  = 256;
    public static final int BLOCKS = (int) ((SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    static final int N = 10;
    static final int REPS = 10;
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

        var startCPU = System.nanoTime();

        doFirstVersionCopies(hPtr, dPtr);

        var stopCPU = System.nanoTime();

        cudaFree(dPtr);
        if (MemoryType.PINNED.equals(memoryTypeUsed)) {
            cudaFreeHost(hPtr);
        }

        System.out.printf("Version 1 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
    }

    private void doFirstVersionCopies(Pointer hPointer, Pointer deviceData) {
        for (int rep = 0; rep < REPS; rep++) {

            cudaMemcpy(deviceData, hPointer, SIZE, cudaMemcpyHostToDevice);
            for (int i = 0; i < N; i++) {
                launchAddOneKernel(deviceData);
            }
            cudaMemcpy(hPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);

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

        System.out.printf("Version 2 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
    }

    private void doSecondVersionCopies(Pointer hostPointer, Pointer deviceData) {
        for (int rep = 0; rep < REPS; rep++) {
            for (int i = 0; i < N; i++) {
                cudaMemcpy(deviceData, hostPointer, SIZE, cudaMemcpyHostToDevice);
                launchAddOneKernel(deviceData);
                cudaMemcpy(hostPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);
            }

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
}


