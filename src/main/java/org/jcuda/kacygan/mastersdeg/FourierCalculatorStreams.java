package org.jcuda.kacygan.mastersdeg;

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

import static jcuda.driver.JCudaDriver.*;

@SuppressWarnings("java:S106")
public class FourierCalculatorStreams implements FourierTest {
    //Fourier: streams version
    private static final String PTX_FILENAME = "FourierRaw.ptx";

    @Override
    public void runTest() {
        System.out.println("TESTING FOURIER WITH STREAMS");

        System.out.println("Performing cold run to warm up GPU...");
        performColdRun();
        System.out.println("Cold run completed.\n");
        var device = new CUdevice();
        cuDeviceGet(device, 0);

        var context = new CUcontext();
        cuCtxCreate(context, 0, device);
        
        var startWholeTime = System.nanoTime();
        for (var rep = 0; rep < NUM_REPS; rep++) {

            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);
            
            var streams = new CUstream[NUM_STREAMS];
            for (var i = 0; i < NUM_STREAMS; i++) {
                streams[i] = new CUstream();
                cuStreamCreate(streams[i], 0);
            }

            var chunkSize = (LENGTH + NUM_STREAMS - 1) / NUM_STREAMS;
            var deviceResults = new CUdeviceptr[NUM_STREAMS];
            var hostResultPtrs = new Pointer[NUM_STREAMS];
            var hostResultBuffers = new java.nio.FloatBuffer[NUM_STREAMS];
            
            var module = new CUmodule();
            cuModuleLoad(module, PTX_FILENAME);

            var function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            for (var i = 0; i < NUM_STREAMS; i++) {
                var startIdx = i * chunkSize;
                var currentChunkSize = Math.min(chunkSize, LENGTH - startIdx);

                deviceResults[i] = new CUdeviceptr();
                cuMemAlloc(deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
                hostResultPtrs[i] = new Pointer();
                cuMemHostAlloc(hostResultPtrs[i], (long)currentChunkSize * Sizeof.FLOAT, 0);
                hostResultBuffers[i] = hostResultPtrs[i].getByteBuffer(0, (long)currentChunkSize * Sizeof.FLOAT).asFloatBuffer();

                var chunkTmin = TMIN + startIdx * DELTA;
                var kernelParameters = Pointer.to(
                    Pointer.to(new float[]{chunkTmin}),
                    Pointer.to(new float[]{DELTA}),
                    Pointer.to(new int[]{LENGTH}),
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
            }
            
            for (var stream : streams) {
                cuStreamSynchronize(stream);
            }

            for (var i = 0; i < NUM_STREAMS; i++) {
                cuMemcpyDtoH(hostResultPtrs[i], deviceResults[i], (long)Math.min(chunkSize, LENGTH - i * chunkSize) * Sizeof.FLOAT);
            }


            for (var i = 0; i < NUM_STREAMS; i++) {
                cuMemFree(deviceResults[i]);
                cuMemFreeHost(hostResultPtrs[i]);
                cuStreamDestroy(streams[i]);
            }
            
        }

        cuCtxDestroy(context);
        
        var endWholeTime = System.nanoTime();

        System.out.printf("  Whole time taken for %d reps: %.6f s\n", NUM_REPS, (endWholeTime - startWholeTime) / 1e9);
    }
    
    private void performColdRun() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        var device = new CUdevice();
        cuDeviceGet(device, 0);

        var context = new CUcontext();
        cuCtxCreate(context, 0, device);
        
        var streams = new CUstream[NUM_STREAMS];
        var chunkSize = (LENGTH + NUM_STREAMS - 1) / NUM_STREAMS;
        var deviceResults = new CUdeviceptr[NUM_STREAMS];
        var hostResultPtrs = new Pointer[NUM_STREAMS];
        var hostResultBuffers = new java.nio.FloatBuffer[NUM_STREAMS];
        for (var i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], 0);
            deviceResults[i] = new CUdeviceptr();
            cuMemAlloc(deviceResults[i], (long)chunkSize * Sizeof.FLOAT);
            hostResultPtrs[i] = new Pointer();
            cuMemHostAlloc(hostResultPtrs[i], (long)chunkSize * Sizeof.FLOAT, 0);
            hostResultBuffers[i] = hostResultPtrs[i].getByteBuffer(0, (long)chunkSize * Sizeof.FLOAT).asFloatBuffer();
        }
        
        var module = new CUmodule();
        cuModuleLoad(module, PTX_FILENAME);

        var function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        for (var i = 0; i < NUM_STREAMS; i++) {
            var startIdx = i * chunkSize;
            var currentChunkSize = Math.min(chunkSize, LENGTH - startIdx);

            var chunkTmin = TMIN + startIdx * DELTA;
            var kernelParameters = Pointer.to(
                Pointer.to(new float[]{chunkTmin}),
                Pointer.to(new float[]{DELTA}),
                Pointer.to(new int[]{LENGTH}),
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
            cuMemcpyDtoH(hostResultPtrs[i], deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
        }
        
        for (var stream : streams) {
            cuStreamSynchronize(stream);
        }

        for (var i = 0; i < NUM_STREAMS; i++) {
            cuMemFree(deviceResults[i]);
            cuMemFreeHost(hostResultPtrs[i]);
            cuStreamDestroy(streams[i]);
        }
        
        cuCtxDestroy(context);
    }
}
