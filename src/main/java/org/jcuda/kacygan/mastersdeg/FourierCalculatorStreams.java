package org.jcuda.kacygan.mastersdeg;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;
import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

@SuppressWarnings("java:S106")
public class FourierCalculatorStreams implements FourierTest {
    private static final String PTX_FILENAME = "FourierRaw.ptx";

    private CUcontext context;
    private CUdevice device;
    private CUmodule module;
    private CUfunction function;
    private CUstream[] streams;
    private CUdeviceptr[] deviceResults;
    private Pointer[] hostResultPointers;
    private java.nio.FloatBuffer[] hostResultBuffers;
    private int chunkSize;
    private boolean resourcesInitialized = false;

    private void initializeResources() {
        if (resourcesInitialized) return;

        device = new CUdevice();
        cuDeviceGet(device, 0);

        context = new CUcontext();
        cuCtxCreate(context, 0, device);

        module = new CUmodule();
        cuModuleLoad(module, PTX_FILENAME);

        function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        chunkSize = (LENGTH + NUM_STREAMS - 1) / NUM_STREAMS;

        streams = new CUstream[NUM_STREAMS];
        deviceResults = new CUdeviceptr[NUM_STREAMS];
        hostResultPointers = new Pointer[NUM_STREAMS];
        hostResultBuffers = new java.nio.FloatBuffer[NUM_STREAMS];

        for (var i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], 0);

            deviceResults[i] = new CUdeviceptr();
            cuMemAlloc(deviceResults[i], (long) chunkSize * Sizeof.FLOAT);

            hostResultPointers[i] = new Pointer();
            cuMemHostAlloc(hostResultPointers[i], (long) chunkSize * Sizeof.FLOAT, 0);

            hostResultBuffers[i] = hostResultPointers[i]
                    .getByteBuffer(0, (long) chunkSize * Sizeof.FLOAT)
                    .asFloatBuffer();
        }

        resourcesInitialized = true;
    }

    private void cleanupResources() {
        if (!resourcesInitialized) return;

        for (var i = 0; i < NUM_STREAMS; i++) {
            if (deviceResults[i] != null) cuMemFree(deviceResults[i]);
            if (hostResultPointers[i] != null) cuMemFreeHost(hostResultPointers[i]);
            if (streams[i] != null) cuStreamDestroy(streams[i]);
        }

        if (module != null) cuModuleUnload(module);
        if (context != null) cuCtxDestroy(context);

        resourcesInitialized = false;
    }

    @Override
    public void runTest() {
        System.out.println("TESTING FOURIER WITH STREAMS");

        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        System.out.println("Performing cold run to warm up GPU...");
        performColdRun();
        System.out.println("Cold run completed.\n");

        initializeResources();

        var startWholeTime = System.nanoTime();
        for (var rep = 0; rep < NUM_REPS; rep++) {
            runSingleTest();
        }
        var endWholeTime = System.nanoTime();

        System.out.printf("  Whole time taken for %d reps: %.3f s\n",
                NUM_REPS, (endWholeTime - startWholeTime) / 1e9);

        cleanupResources();
    }

    private void runSingleTest() {
        for (var i = 0; i < NUM_STREAMS; i++) {
            var startIdx = i * chunkSize;
            var currentChunkSize = Math.min(chunkSize, LENGTH - startIdx);

            var chunkTmin = TMIN + startIdx * DELTA;

            var kernelParameters = Pointer.to(
                    Pointer.to(new float[]{chunkTmin}),
                    Pointer.to(new float[]{DELTA}),
                    Pointer.to(new int[]{currentChunkSize}),
                    Pointer.to(new int[]{COEFFICIENTS}),
                    Pointer.to(new float[]{PI}),
                    Pointer.to(new float[]{PI_OVER_T}),
                    Pointer.to(new float[]{RESULT_COEFFICIENT}),
                    Pointer.to(new float[]{PERIOD}),
                    Pointer.to(deviceResults[i])
            );

            var blocksPerGrid = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            cuLaunchKernel(function,
                    blocksPerGrid, 1, 1,
                    THREADS_PER_BLOCK, 1, 1,
                    0, streams[i],
                    kernelParameters, null
            );

            cuMemcpyDtoHAsync(hostResultPointers[i], deviceResults[i],
                    (long) currentChunkSize * Sizeof.FLOAT, streams[i]);
        }

        for (var stream : streams) {
            cuStreamSynchronize(stream);
        }
    }

    private void performColdRun() {
        var device = new CUdevice();
        cuDeviceGet(device, 0);

        var context = new CUcontext();
        cuCtxCreate(context, 0, device);

        var module = new CUmodule();
        cuModuleLoad(module, PTX_FILENAME);

        var function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        var deviceResult = new CUdeviceptr();
        cuMemAlloc(deviceResult, (long) LENGTH * Sizeof.FLOAT);

        var kernelParameters = Pointer.to(
                Pointer.to(new float[]{TMIN}),
                Pointer.to(new float[]{DELTA}),
                Pointer.to(new int[]{LENGTH}),
                Pointer.to(new int[]{COEFFICIENTS}),
                Pointer.to(new float[]{PI}),
                Pointer.to(new float[]{PI_OVER_T}),
                Pointer.to(new float[]{RESULT_COEFFICIENT}),
                Pointer.to(new float[]{PERIOD}),
                Pointer.to(deviceResult)
        );

        var blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                0, null,
                kernelParameters, null
        );

        cuStreamSynchronize(null);

        cuMemFree(deviceResult);
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }
}