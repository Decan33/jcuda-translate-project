package org.jcuda.kacygan.chapter10;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

import java.util.Arrays;
import java.util.Random;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT;
import static jcuda.driver.CUevent_flags.CU_EVENT_DEFAULT;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoDAsync;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

public class BasicDoubleStreamCorrect {
    private static final int N = 1024 * 1024;
    private static final int FULL_DATA_SIZE = N * 20;

    private static final String KERNEL_CODE =
            "extern \"C\" __global__ void kernel(int *a, int *b, int *c) {" +
                    "    int idx = threadIdx.x + blockIdx.x * blockDim.x;" +
                    "    if (idx < " + N + ") {" +
                    "        int idx1 = (idx + 1) % 256;" +
                    "        int idx2 = (idx + 2) % 256;" +
                    "        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;" +
                    "        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;" +
                    "        c[idx] = (int)((as + bs) / 2);" +
                    "    }" +
                    "}";

    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var random = new Random();

        var device = new CUdevice();
        var context = new CUcontext();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        var deviceOverlap = new int[1];
        cuDeviceGetAttribute(deviceOverlap, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device);
        if (deviceOverlap[0] == 0) {
            System.out.println("Device will not handle overlaps, so no speed up from streams");
            return;
        }

        var start = new CUevent();
        var stop = new CUevent();
        cuEventCreate(start, CU_EVENT_DEFAULT);
        cuEventCreate(stop, CU_EVENT_DEFAULT);
        var stream0 = new CUstream();
        var stream1 = new CUstream();
        cuStreamCreate(stream0, 0);
        cuStreamCreate(stream1, 0);

        var hostA = new int[FULL_DATA_SIZE];
        var hostB = new int[FULL_DATA_SIZE];
        var hostC = new int[FULL_DATA_SIZE];
        for (int i = 0; i < FULL_DATA_SIZE; i++) {
            hostA[i] = (random.nextInt() * Integer.MAX_VALUE);
            hostB[i] = (random.nextInt() * Integer.MAX_VALUE);
        }

        var devA0 = new CUdeviceptr();
        var devB0 = new CUdeviceptr();
        var devC0 = new CUdeviceptr();
        cuMemAlloc(devA0, N * Sizeof.INT);
        cuMemAlloc(devB0, N * Sizeof.INT);
        cuMemAlloc(devC0, N * Sizeof.INT);

        var devA1 = new CUdeviceptr();
        var devB1 = new CUdeviceptr();
        var devC1 = new CUdeviceptr();
        cuMemAlloc(devA1, N * Sizeof.INT);
        cuMemAlloc(devB1, N * Sizeof.INT);
        cuMemAlloc(devC1, N * Sizeof.INT);

// Load the PTX containing the compiled kernel function
        var module = new CUmodule();
        cuModuleLoad(module, "kernel.ptx");
        var kernelFunction = new CUfunction();
        cuModuleGetFunction(kernelFunction, module, "kernel");

        cuEventRecord(start, null);
        for (int i = 0; i < FULL_DATA_SIZE; i += N * 2) {
            // Create temporary slices of the host arrays for the current segment
            int[] hostASegment0 = Arrays.copyOfRange(hostA, i, i + N);
            int[] hostASegment1 = Arrays.copyOfRange(hostA, i + N, i + 2 * N);
            int[] hostBSegment0 = Arrays.copyOfRange(hostB, i, i + N);
            int[] hostBSegment1 = Arrays.copyOfRange(hostB, i + N, i + 2 * N);

            // Copy segments to device memory
            cuMemcpyHtoDAsync(devA0, Pointer.to(hostASegment0), N * Sizeof.INT, stream0);
            cuMemcpyHtoDAsync(devA1, Pointer.to(hostASegment1), N * Sizeof.INT, stream1);
            cuMemcpyHtoDAsync(devB0, Pointer.to(hostBSegment0), N * Sizeof.INT, stream0);
            cuMemcpyHtoDAsync(devB1, Pointer.to(hostBSegment1), N * Sizeof.INT, stream1);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(new int[]{N}),
                    Pointer.to(devA0),
                    Pointer.to(devB0),
                    Pointer.to(devC0)
            );

            // Kernel launch parameters: grid size and block size
            int blockSize = 256; // Example block size
            int gridSize = (N + blockSize - 1) / blockSize;

            // Launch the kernel
            cuLaunchKernel(kernelFunction,
                    gridSize, 1, 1,      // Grid dimension
                    blockSize, 1, 1,   // Block dimension
                    0, stream0,         // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            // Repeat kernel execution for the second segment (devA1, devB1, devC1, stream1)
            Pointer kernelParameters1 = Pointer.to(
                    Pointer.to(new int[]{N}),
                    Pointer.to(devA1),
                    Pointer.to(devB1),
                    Pointer.to(devC1)
            );

            cuLaunchKernel(kernelFunction,
                    gridSize, 1, 1,    // Grid dimension
                    blockSize, 1, 1, // Block dimension
                    0, stream1,       // Shared memory size and stream
                    kernelParameters1, null // Kernel- and extra parameters
            );


            // Since we cannot directly write back to a segment of the hostC array,
            // we'll need to handle the resulting segments individually after copying back.
            int[] hostCSegment0 = new int[N];
            int[] hostCSegment1 = new int[N];
            cuMemcpyDtoHAsync(Pointer.to(hostCSegment0), devC0, N * Sizeof.INT, stream0);
            cuMemcpyDtoHAsync(Pointer.to(hostCSegment1), devC1, N * Sizeof.INT, stream1);

            // After copying back, integrate the segments back into the main hostC array.
            System.arraycopy(hostCSegment0, 0, hostC, i, N);
            System.arraycopy(hostCSegment1, 0, hostC, i + N, N);
        }
        cuStreamSynchronize(stream0);
        cuStreamSynchronize(stream1);
        cuEventRecord(stop, null);
        cuEventSynchronize(stop);

        float[] elapsedTime = new float[1];
        cuEventElapsedTime(elapsedTime, start, stop);
        System.out.printf("Time taken: %3.1f ms\n", elapsedTime[0]);

        cuMemFree(devA0);
        cuMemFree(devB0);
        cuMemFree(devC0);
        cuMemFree(devA1);
        cuMemFree(devB1);
        cuMemFree(devC1);
        cuStreamDestroy(stream0);
        cuStreamDestroy(stream1);
        cuCtxDestroy(context);
    }
}

