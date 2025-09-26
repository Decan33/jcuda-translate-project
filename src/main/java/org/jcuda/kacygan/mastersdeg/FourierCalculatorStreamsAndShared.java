package org.jcuda.kacygan.mastersdeg;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import static jcuda.driver.JCudaDriver.*;

@SuppressWarnings("java:S106")
public class FourierCalculatorStreamsAndShared implements FourierTest {

    private static final String PTX_FILENAME = "FourierOptimized.ptx";
    private static final int CHUNK_SIZE = (LENGTH + NUM_STREAMS - 1) / NUM_STREAMS;

    private CUcontext context;
    private CUstream[] streams;
    private CUmodule[] modules;
    private CUfunction[] functions;
    private CUdeviceptr[] deviceChunks;
    private Pointer[] hostChunkPointers;
    private FloatBuffer[] hostBuffers;
    private int[] chunkSizes;
    private boolean resourcesInitialized = false;

    private void initializeResources() {
        if (resourcesInitialized) return;

        context = initJCuda();

        streams = new CUstream[NUM_STREAMS];
        modules = new CUmodule[NUM_STREAMS];
        functions = new CUfunction[NUM_STREAMS];
        deviceChunks = new CUdeviceptr[NUM_STREAMS];
        hostChunkPointers = new Pointer[NUM_STREAMS];
        hostBuffers = new FloatBuffer[NUM_STREAMS];
        chunkSizes = new int[NUM_STREAMS];

        for (int i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], 0);

            modules[i] = new CUmodule();
            cuModuleLoad(modules[i], PTX_FILENAME);

            functions[i] = new CUfunction();
            cuModuleGetFunction(functions[i], modules[i], FUNCTION_NAME);

            int startIdx = i * CHUNK_SIZE;
            int endIdx = Math.min(startIdx + CHUNK_SIZE, LENGTH);
            int currentChunkSize = Math.max(0, endIdx - startIdx);
            chunkSizes[i] = currentChunkSize;

            if (currentChunkSize > 0) {
                deviceChunks[i] = new CUdeviceptr();
                cuMemAlloc(deviceChunks[i], (long) currentChunkSize * Sizeof.FLOAT);

                hostChunkPointers[i] = new Pointer();
                cuMemHostAlloc(hostChunkPointers[i], (long) currentChunkSize * Sizeof.FLOAT,
                        JCudaDriver.CU_MEMHOSTALLOC_PORTABLE);
                ByteBuffer bb = hostChunkPointers[i].getByteBuffer(0, (long) currentChunkSize * Sizeof.FLOAT)
                        .order(ByteOrder.nativeOrder());
                hostBuffers[i] = bb.asFloatBuffer();
            }
        }

        resourcesInitialized = true;
    }

    private void cleanupResources() {
        if (!resourcesInitialized) return;

        for (int i = 0; i < NUM_STREAMS; i++) {
            if (deviceChunks[i] != null) cuMemFree(deviceChunks[i]);
            if (hostChunkPointers[i] != null) cuMemFreeHost(hostChunkPointers[i]);
            if (streams[i] != null) cuStreamDestroy(streams[i]);
            if (modules[i] != null) cuModuleUnload(modules[i]);
        }

        if (context != null) cuCtxDestroy(context);
        resourcesInitialized = false;
    }

    @Override
    public void runTest() {
        System.out.println("TESTING FOURIER WITH SHARED, CONSTANTS AND STREAMS");

        System.out.println("Performing cold run to warm up GPU...");
        performColdRun();
        System.out.println("Cold run completed.\n");

        initializeResources();

        long startWholeTime = System.nanoTime();
        for (int rep = 0; rep < NUM_REPS; rep++) {
            runSingleTest();
        }
        long endWholeTime = System.nanoTime();

        double wholeTimeSec = (endWholeTime - startWholeTime) / 1e9;

        System.out.println("\n=========================");
        System.out.printf("Whole time taken for %d reps: %.6f s%n", NUM_REPS, wholeTimeSec);
        System.out.println("=========================");

        cleanupResources();
    }

    private void runSingleTest() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            int currentChunkSize = chunkSizes[i];
            if (currentChunkSize == 0) continue;

            int startIdx = i * CHUNK_SIZE;
            float tminForChunk = TMIN + startIdx * DELTA;

            setAllConstants(modules[i], tminForChunk, streams[i], COEFFICIENTS);

            Pointer kernelParams = Pointer.to(Pointer.to(deviceChunks[i]));
            int blocks = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            cuLaunchKernel(functions[i],
                    blocks, 1, 1,
                    THREADS_PER_BLOCK, 1, 1,
                    0,
                    streams[i],
                    kernelParams, null);

            cuMemcpyDtoHAsync(hostChunkPointers[i], deviceChunks[i],
                    (long) currentChunkSize * Sizeof.FLOAT, streams[i]);
        }

        for (CUstream stream : streams) {
            cuStreamSynchronize(stream);
        }
    }


    private static void setAllConstants(CUmodule modules, float tminForChunk, CUstream streams, int coefficients) {
        setConstant(modules, "const_tmin", Pointer.to(new float[]{tminForChunk}), Sizeof.FLOAT, streams);
        setConstant(modules, "const_delta", Pointer.to(new float[]{DELTA}), Sizeof.FLOAT, streams);
        setConstant(modules, "const_coefficients", Pointer.to(new int[]{coefficients}), Sizeof.INT, streams);
        setConstant(modules, "const_pi", Pointer.to(new float[]{PI}), Sizeof.FLOAT, streams);
        setConstant(modules, "const_pi_squared", Pointer.to(new float[]{PI_SQUARED}), Sizeof.FLOAT, streams);
        setConstant(modules, "const_T", Pointer.to(new float[]{PERIOD}), Sizeof.FLOAT, streams);
        setConstant(modules, "const_pi_over_T", Pointer.to(new float[]{PI_OVER_T}), Sizeof.FLOAT, streams);
        setConstant(modules, "constant_result_coefficient", Pointer.to(new float[]{RESULT_COEFFICIENT}), Sizeof.FLOAT, streams);
    }

    private static CUcontext initJCuda() {
        JCuda.setExceptionsEnabled(true);
        JCudaDriver.setExceptionsEnabled(true);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
        return context;
    }

    private static void setConstant(CUmodule module, String symbolName, Pointer hostData, long sizeBytes, CUstream stream) {
        CUdeviceptr constPtr = new CUdeviceptr();
        long[] symbolSize = new long[1];
        cuModuleGetGlobal(constPtr, symbolSize, module, symbolName);
        cuMemcpyHtoDAsync(constPtr, hostData, sizeBytes, stream);
    }

    private static void performColdRun() {
        CUcontext context = initJCuda();

        CUstream stream = new CUstream();
        cuStreamCreate(stream, 0);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, PTX_FILENAME);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "fourier");

        CUdeviceptr dOut = new CUdeviceptr();
        cuMemAlloc(dOut, (long) LENGTH * Sizeof.FLOAT);

        setAllConstants(module, TMIN, stream, Math.min(COEFFICIENTS, 1024));

        Pointer kernelParams = Pointer.to(Pointer.to(dOut));
        int blocks = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cuLaunchKernel(function,
                blocks, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                0, stream,
                kernelParams, null);
        cuStreamSynchronize(stream);

        cuMemFree(dOut);
        cuModuleUnload(module);
        cuStreamDestroy(stream);
        cuCtxDestroy(context);
    }
}