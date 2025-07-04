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
public class FourierCalculator4 implements FourierTest {
    //Fourier: streams version
    private static final String PTX_FILENAME = "Fourier3.ptx";
    private static final int NUM_STREAMS = 4;

    @Override
    public void runTest() {
        // Cold run to warm up GPU
        System.out.println("Performing cold run to warm up GPU...");
        performColdRun();
        System.out.println("Cold run completed.\n");
        
        var prepTimes = new double[NUM_REPS];
        var kernelTimes = new double[NUM_REPS];
        var deleteTimes = new double[NUM_REPS];
        
        var startWholeTime = System.nanoTime();

        for (var rep = 0; rep < NUM_REPS; rep++) {
            var prepStart = System.nanoTime();
            
            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);

            var device = new CUdevice();
            cuDeviceGet(device, 0);

            var context = new CUcontext();
            cuCtxCreate(context, 0, device);

            var delta = (TMAX - TMIN) / (LENGTH - 1);
            
            var streams = new CUstream[NUM_STREAMS];
            for (var i = 0; i < NUM_STREAMS; i++) {
                streams[i] = new CUstream();
                cuStreamCreate(streams[i], 0);
            }

            var chunkSize = (LENGTH + NUM_STREAMS - 1) / NUM_STREAMS;
            var deviceResults = new CUdeviceptr[NUM_STREAMS];
            var hostResults = new float[NUM_STREAMS][];
            
            var module = new CUmodule();
            cuModuleLoad(module, PTX_FILENAME);

            var function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            var prepEnd = System.nanoTime();
            prepTimes[rep] = (prepEnd - prepStart) / 1e9;
            
            var kernelStart = new CUevent();
            var kernelStop = new CUevent();

            cuEventCreate(kernelStart, 0);
            cuEventCreate(kernelStop, 0);
            cuEventRecord(kernelStart, null);

            for (var i = 0; i < NUM_STREAMS; i++) {
                var startIdx = i * chunkSize;
                var currentChunkSize = Math.min(chunkSize, LENGTH - startIdx);

                deviceResults[i] = new CUdeviceptr();
                cuMemAlloc(deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
                hostResults[i] = new float[currentChunkSize];

                var chunkTmin = TMIN + startIdx * delta;
                float pi = (float)Math.PI;
                float T = TMAX - TMIN;
                float pi_over_T = pi / T;
                float result_coefficient = 4.0f / (pi * pi);
                var kernelParameters = Pointer.to(
                    Pointer.to(new float[]{chunkTmin}),
                    Pointer.to(new float[]{delta}),
                    Pointer.to(new int[]{LENGTH}),
                    Pointer.to(new int[]{COEFFICIENTS}),
                    Pointer.to(new float[]{pi}),
                    Pointer.to(new float[]{pi_over_T}),
                    Pointer.to(new float[]{result_coefficient}),
                    Pointer.to(new float[]{T}),
                    Pointer.to(deviceResults[i]),
                    Pointer.to(new int[]{startIdx}),
                    Pointer.to(new int[]{currentChunkSize})
                );

                var blocksPerGrid = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                cuLaunchKernel(function,
                    blocksPerGrid, 1, 1,
                    THREADS_PER_BLOCK, 1, 1,
                    0, streams[i],
                    kernelParameters, null
                );

                cuMemcpyDtoH(Pointer.to(hostResults[i]), deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
            }
            
            for (var stream : streams) {
                cuStreamSynchronize(stream);
            }

            cuEventRecord(kernelStop, null);
            cuEventSynchronize(kernelStop);
            var kernelMs = new float[1];

            cuEventElapsedTime(kernelMs, kernelStart, kernelStop);
            kernelTimes[rep] = kernelMs[0] / 1000.0;
            
            cuEventDestroy(kernelStart);
            cuEventDestroy(kernelStop);

            var deleteStart = System.nanoTime();
            
            for (var i = 0; i < NUM_STREAMS; i++) {
                cuMemFree(deviceResults[i]);
                cuStreamDestroy(streams[i]);
            }

            cuCtxDestroy(context);
            
            var deleteEnd = System.nanoTime();
            deleteTimes[rep] = (deleteEnd - deleteStart) / 1e9;
        }
        
        var endWholeTime = System.nanoTime();

        logTimings(prepTimes, kernelTimes, deleteTimes, endWholeTime - startWholeTime);
    }

    private void logTimings(double[] prep, double[] kernel, double[] del, double wholeTime) {
        for (var i = 0; i < prep.length; i++) {
            System.out.printf("Repetition %d:\n", i + 1);
            System.out.printf("  Preparation time: %.6f s\n", prep[i]);
            System.out.printf("  Kernel execution time: %.6f s\n", kernel[i]);
            System.out.printf("  Memory deletion time: %.6f s\n", del[i]);
        }
        var n = prep.length;
        var prepAvg = mean(prep);
        var kernelAvg = mean(kernel);
        var delAvg = mean(del);
        var prepStd = standardDeviation(prep, prepAvg);
        var kernelStd = standardDeviation(kernel, kernelAvg);
        var delStd = standardDeviation(del, delAvg);
        System.out.printf("\nAverages over %d repetitions:\n", n);
        System.out.printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prepAvg, prepStd);
        System.out.printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernelAvg, kernelStd);
        System.out.printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delAvg, delStd);
        System.out.printf("  Whole time taken for %d reps: %.6f s\n", NUM_REPS, wholeTime / 1e9);
        System.out.println("=========================");
    }

    private double mean(double[] arr) {
        var sum = 0.0;
        for (var v : arr) sum += v;
        return sum / arr.length;
    }

    private double standardDeviation(double[] arr, double mean) {
        var sum = 0.0;
        for (var v : arr) sum += (v - mean) * (v - mean);
        return Math.sqrt(sum / arr.length);
    }
    
    private void performColdRun() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        var device = new CUdevice();
        cuDeviceGet(device, 0);

        var context = new CUcontext();
        cuCtxCreate(context, 0, device);

        var delta = (TMAX - TMIN) / (LENGTH - 1);
        
        var streams = new CUstream[NUM_STREAMS];
        for (var i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], 0);
        }

        var chunkSize = (LENGTH + NUM_STREAMS - 1) / NUM_STREAMS;
        var deviceResults = new CUdeviceptr[NUM_STREAMS];
        var hostResults = new float[NUM_STREAMS][];
        
        var module = new CUmodule();
        cuModuleLoad(module, PTX_FILENAME);

        var function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        for (var i = 0; i < NUM_STREAMS; i++) {
            var startIdx = i * chunkSize;
            var currentChunkSize = Math.min(chunkSize, LENGTH - startIdx);

            deviceResults[i] = new CUdeviceptr();
            cuMemAlloc(deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
            hostResults[i] = new float[currentChunkSize];

            var chunkTmin = TMIN + startIdx * delta;
            float pi = (float)Math.PI;
            float T = TMAX - TMIN;
            float pi_over_T = pi / T;
            float result_coefficient = 4.0f / (pi * pi);
            var kernelParameters = Pointer.to(
                Pointer.to(new float[]{chunkTmin}),
                Pointer.to(new float[]{delta}),
                Pointer.to(new int[]{LENGTH}),
                Pointer.to(new int[]{COEFFICIENTS}),
                Pointer.to(new float[]{pi}),
                Pointer.to(new float[]{pi_over_T}),
                Pointer.to(new float[]{result_coefficient}),
                Pointer.to(new float[]{T}),
                Pointer.to(deviceResults[i]),
                Pointer.to(new int[]{startIdx}),
                Pointer.to(new int[]{currentChunkSize})
            );

            var blocksPerGrid = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                0, streams[i],
                kernelParameters, null
            );

            cuMemcpyDtoH(Pointer.to(hostResults[i]), deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
        }
        
        for (var stream : streams) {
            cuStreamSynchronize(stream);
        }

        // Cleanup
        for (var i = 0; i < NUM_STREAMS; i++) {
            cuMemFree(deviceResults[i]);
            cuStreamDestroy(streams[i]);
        }
        
        cuCtxDestroy(context);
    }
}
