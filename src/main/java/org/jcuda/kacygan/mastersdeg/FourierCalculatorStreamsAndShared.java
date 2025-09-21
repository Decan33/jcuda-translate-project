package org.jcuda.kacygan.mastersdeg;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import static jcuda.driver.JCudaDriver.*;

/**
 * Streams + async version that leverages constant memory (for scalar parameters)
 * and the kernel's shared memory usage. We DO NOT modify the CUDA kernel.
 *
 * Key points:
 *  - Loads one CUmodule per stream so each stream can set its own constant symbols safely.
 *  - Pushes scalar parameters into __constant__ symbols via cuModuleGetGlobal + cuMemcpyHtoDAsync.
 *  - Uses page-locked (pinned) host buffers and cuMemcpyDtoHAsync for overlapped copies.
 *  - Launches kernels in parallel streams and synchronizes at the end.
 *  - Prefers shared memory cache configuration on the kernel.
 */
@SuppressWarnings("java:S106")
public class FourierCalculatorStreamsAndShared implements FourierTest {

    // ======== Configuration (adjust to your project’s constants if needed) ========
    private static final String PTX_FILENAME = "FourierOptimized.ptx";

    // Derived chunking
    private static final int    CHUNK_SIZE = (LENGTH + NUM_STREAMS - 1) / NUM_STREAMS;

    @Override
    public void runTest() {
        System.out.println("TESTING FOURIER WITH SHARED, CONSTANTS AND STREAMS");

        // Warm-up to mitigate first-use overheads
        System.out.println("Performing cold run to warm up GPU...");
        performColdRun();
        System.out.println("Cold run completed.\n");

        CUcontext context = initJCuda();

        long startWholeTime = System.nanoTime();
        for (int rep = 0; rep < NUM_REPS; rep++) {

            CUstream[]     streams       = new CUstream[NUM_STREAMS];
            CUmodule[]     modules       = new CUmodule[NUM_STREAMS];
            CUfunction[]   functions     = new CUfunction[NUM_STREAMS];
            CUdeviceptr[]  deviceChunks  = new CUdeviceptr[NUM_STREAMS];
            Pointer[]      hostChunkPtrs = new Pointer[NUM_STREAMS];
            FloatBuffer[]  hostBuffers   = new FloatBuffer[NUM_STREAMS];
            int[]          chunkSizes    = new int[NUM_STREAMS];

            prepareDataAndAllocateMemory(streams, modules, functions, deviceChunks, hostChunkPtrs, hostBuffers, chunkSizes);

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

                cuMemcpyDtoHAsync(hostChunkPtrs[i], deviceChunks[i], (long) currentChunkSize * Sizeof.FLOAT, streams[i]);
            }

            for (CUstream stream : streams) {
                cuStreamSynchronize(stream);
            }

            for (int i = 0; i < NUM_STREAMS; i++) {
                if (deviceChunks[i] != null)  cuMemFree(deviceChunks[i]);
                if (hostChunkPtrs[i] != null) cuMemFreeHost(hostChunkPtrs[i]);
                if (streams[i] != null)       cuStreamDestroy(streams[i]);
                if (modules[i] != null)       cuModuleUnload(modules[i]);
            }
        }

        long endWholeTime = System.nanoTime();
        double wholeTimeSec = (endWholeTime - startWholeTime) / 1e9;

        // Report averages
        System.out.println("\n=========================");
        System.out.printf("Whole time taken for %d reps: %.6f s%n", NUM_REPS, wholeTimeSec);
        System.out.println("=========================");

        cuCtxDestroy(context);
    }

    private static void prepareDataAndAllocateMemory(CUstream[] streams, CUmodule[] modules, CUfunction[] functions, CUdeviceptr[] deviceChunks, Pointer[] hostChunkPtrs, FloatBuffer[] hostBuffers, int[] chunkSizes) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], 0);

            modules[i] = new CUmodule();
            cuModuleLoad(modules[i], PTX_FILENAME);

            functions[i] = new CUfunction();
            cuModuleGetFunction(functions[i], modules[i], "fourier");

            int startIdx = i * CHUNK_SIZE;
            int endIdx = Math.min(startIdx + CHUNK_SIZE, LENGTH);
            int currentChunkSize = Math.max(0, endIdx - startIdx);
            chunkSizes[i] = currentChunkSize;

            deviceChunks[i] = new CUdeviceptr();
            cuMemAlloc(deviceChunks[i], (long) currentChunkSize * Sizeof.FLOAT);

            hostChunkPtrs[i] = new Pointer();
            cuMemHostAlloc(hostChunkPtrs[i], (long) currentChunkSize * Sizeof.FLOAT, JCudaDriver.CU_MEMHOSTALLOC_PORTABLE);
            ByteBuffer bb = hostChunkPtrs[i].getByteBuffer(0, (long) currentChunkSize * Sizeof.FLOAT).order(ByteOrder.nativeOrder());
            hostBuffers[i] = bb.asFloatBuffer();
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
