package org.jcuda.kacygan.mastersdeg;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

public class JCudaMemoryTest {

    static final long SIZE = 1L * 1024 * 1024 * 1024;
    static final Integer THREADS_PER_BLOCK  = 256;
    static final int N = 10;
    static final int REPS = 10;
    static final MemoryType memoryTypeUsed = MemoryType.PINNED;

    private enum MemoryType {
        PINNED,
        NORMAL
    }

    public static void main(String[] args) {
        initJCuda();

        for (int i = 0; i < 5; i++) {
            version1();
            version2();
        }
    }

    static void initJCuda() {
        JCuda.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        var context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }

    static void version1() {
        var startCPU = System.nanoTime();
        if (MemoryType.NORMAL.equals(memoryTypeUsed)) {
            for (int rep = 0; rep < REPS; rep++) {
                var hostData = new byte[(int) SIZE];
                Arrays.fill(hostData, (byte) 0);        //Same as memset(hostData, 0)
                var hPointer = Pointer.to(hostData);
                var deviceData = new Pointer();
                cudaMalloc(deviceData, SIZE);

                cudaMemcpy(deviceData, hPointer, SIZE, cudaMemcpyHostToDevice);
                for (int i = 0; i < N; i++) {
                    launchAddOneKernel(deviceData);
                }

                cudaMemcpy(hPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);

                cudaFree(deviceData);
                cudaFree(hPointer);     // Same as free(h_data)
            }
        } else {
            //We use pinned memory
            for (int rep = 0; rep < REPS; rep++) {
                var hostData = new byte[(int) SIZE];
                Arrays.fill(hostData, (byte) 0);    //Same as memset(hostData, 0)
                var hPointer = Pointer.to(hostData);
                var deviceData = new Pointer();
                cudaHostAlloc(deviceData, SIZE, cudaHostAllocDefault);

                cudaMemcpy(deviceData, hPointer, SIZE, cudaMemcpyHostToDevice);
                for (int i = 0; i < N; i++) {
                    launchAddOneKernel(deviceData);
                }

                cudaMemcpy(hPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);

                cudaFreeHost(deviceData);
                cudaFree(hPointer);     // Same as free(h_data)
            }
        }

        var stopCPU = System.nanoTime();

        System.out.printf("Version 1 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
    }

    static void version2() {
        var startCPU = System.nanoTime();
        if (MemoryType.NORMAL.equals(memoryTypeUsed)) {
            for (int rep = 0; rep < REPS; rep++) {
                byte[] hostData = new byte[(int) SIZE];
                Arrays.fill(hostData, (byte) 0);        //Same as memset(hostData, 0)
                var hostPointer = Pointer.to(hostData);
                var deviceData = new Pointer();
                cudaMalloc(deviceData, SIZE);

                for (int i = 0; i < N; i++) {
                    cudaMemcpy(deviceData, hostPointer, SIZE, cudaMemcpyHostToDevice);
                    launchAddOneKernel(deviceData);
                    cudaMemcpy(hostPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);
                }

                cudaFree(deviceData);
                cudaFree(hostPointer);
            }
        }else {
            for (int rep = 0; rep < REPS; rep++) {
                byte[] hostData = new byte[(int) SIZE];
                Arrays.fill(hostData, (byte) 0);        //Same as memset(hostData, 0)
                var hostPointer = Pointer.to(hostData);
                var deviceData = new Pointer();
                cudaHostAlloc(deviceData, SIZE, cudaHostAllocDefault);

                for (int i = 0; i < N; i++) {
                    cudaMemcpy(deviceData, hostPointer, SIZE, cudaMemcpyHostToDevice);
                    launchAddOneKernel(deviceData);
                    cudaMemcpy(hostPointer, deviceData, SIZE, cudaMemcpyDeviceToHost);
                }

                cudaFreeHost(deviceData);
                cudaFree(hostPointer);
            }
        }

        var stopCPU = System.nanoTime();

        System.out.printf("Version 2 CPU time: %.3f s\n", (stopCPU - startCPU) / 1e9);
    }

    static void launchAddOneKernel(Pointer data) {
        var blocksPerGrid = (int)((SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

        var module = new CUmodule();
        cuModuleLoad(module, "AddOneKernel.ptx");

        var function = new CUfunction();
        cuModuleGetFunction(function, module, "addOneKernel");

        var kernelParams = Pointer.to(
                Pointer.to(data),
                Pointer.to(new long[]{SIZE})
        );

        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                0, null,
                kernelParams, null);

        cuCtxSynchronize();
    }
}


