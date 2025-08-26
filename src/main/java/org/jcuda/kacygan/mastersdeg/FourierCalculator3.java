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

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventDestroy;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoHAsync;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleGetGlobal;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;
import static jcuda.driver.JCudaDriver.cuMemHostAlloc;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;

@SuppressWarnings("java:S106")
public class FourierCalculator3 implements FourierTest {
    //Fourier: Streams + shared version
    private static final String PTX_FILENAME = "Fourier4.ptx";


    @Override
    public void runTest() {
        System.out.println("TESTING FOURIER WITH SHARED, CONSTANTS AND STREAMS");

        System.out.println("Performing cold run to warm up GPU...");
        performColdRun();
        System.out.println("Cold run completed.\n");
        
        var prepTimes = new double[NUM_REPS];
        var kernelTimes = new double[NUM_REPS];
        var copyTimes = new double[NUM_REPS];
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

            var chunkSize = LENGTH / NUM_STREAMS;
            
            var module = new CUmodule();
            cuModuleLoad(module, PTX_FILENAME);

            var function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            var streams = new CUstream[NUM_STREAMS];
            var deviceChunks = new CUdeviceptr[NUM_STREAMS];
            var hostChunkPtrs = new Pointer[NUM_STREAMS];
            var hostChunkBuffers = new java.nio.FloatBuffer[NUM_STREAMS];
            for (var i = 0; i < NUM_STREAMS; i++) {
                streams[i] = new CUstream();
                cuStreamCreate(streams[i], 0);

                var startIdx = i * chunkSize;
                var endIdx = Math.min(startIdx + chunkSize, LENGTH);
                var currentChunkSize = endIdx - startIdx;
                
                deviceChunks[i] = new CUdeviceptr();
                cuMemAlloc(deviceChunks[i], (long) currentChunkSize * Sizeof.FLOAT);
                hostChunkPtrs[i] = new Pointer();
                cuMemHostAlloc(hostChunkPtrs[i], (long) currentChunkSize * Sizeof.FLOAT, 0);
                hostChunkBuffers[i] = hostChunkPtrs[i].getByteBuffer(0, (long) currentChunkSize * Sizeof.FLOAT).asFloatBuffer();
            }

            var prepEnd = System.nanoTime();
            prepTimes[rep] = (prepEnd - prepStart) / 1e9;

            var kernelStart = new CUevent();
            var kernelStop = new CUevent();
            cuEventCreate(kernelStart, 0);
            cuEventCreate(kernelStop, 0);
            cuEventRecord(kernelStart, null);

            for (var i = 0; i < NUM_STREAMS; i++) {
                var startIdx = i * chunkSize;
                var endIdx = Math.min(startIdx + chunkSize, LENGTH);
                var currentChunkSize = endIdx - startIdx;
                
                setConstant(module, "d_params", new float[]{
                    TMIN + startIdx * DELTA,  // tmin
                    TMAX,                     // tmax
                    LENGTH,                   // length
                    COEFFICIENTS,             // coefficients
                    DELTA                     // delta
                });
                
                var coefficients = new float[COEFFICIENTS];
                for (int k = 0; k < COEFFICIENTS; k++) {
                    coefficients[k] = 1.0f / (4.0f * (k + 1) * (k + 1) - 4.0f * (k + 1) + 1.0f);
                }
                setConstant(module, "d_coefficients", coefficients);
                
                var kernelParams = Pointer.to(
                    Pointer.to(new int[]{startIdx}),
                    Pointer.to(new int[]{endIdx}),
                    Pointer.to(deviceChunks[i])
                );

                var blocks = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                var sharedMemSize = COEFFICIENTS * Sizeof.FLOAT;
                cuLaunchKernel(function,
                    blocks, 1, 1,
                    THREADS_PER_BLOCK, 1, 1,
                    sharedMemSize, streams[i],
                    kernelParams, null);
            }
            
            for (var stream : streams) {
                cuStreamSynchronize(stream);
            }

            cuEventRecord(kernelStop, null);
            cuEventSynchronize(kernelStop);

            var kernelMs = new float[1];
            cuEventElapsedTime(kernelMs, kernelStart, kernelStop);
            kernelTimes[rep] = kernelMs[0] / THOUSAND;

            cuEventDestroy(kernelStart);
            cuEventDestroy(kernelStop);
            
            var copyStart = new CUevent();
            var copyStop = new CUevent();
            cuEventCreate(copyStart, 0);
            cuEventCreate(copyStop, 0);
            cuEventRecord(copyStart, null);
            for (var i = 0; i < NUM_STREAMS; i++) {
                cuMemcpyDtoHAsync(hostChunkPtrs[i], deviceChunks[i], (long) hostChunkBuffers[i].capacity() * Sizeof.FLOAT, streams[i]);
            }
            for (var stream : streams) {
                cuStreamSynchronize(stream);
            }
            cuEventRecord(copyStop, null);
            cuEventSynchronize(copyStop);
            var copyMs = new float[1];
            cuEventElapsedTime(copyMs, copyStart, copyStop);
            copyTimes[rep] = copyMs[0] / 1000.0;
            cuEventDestroy(copyStart);
            cuEventDestroy(copyStop);

            var deleteStart = System.nanoTime();

            for (var i = 0; i < NUM_STREAMS; i++) {
                cuMemFree(deviceChunks[i]);
                cuMemFreeHost(hostChunkPtrs[i]);
                cuStreamDestroy(streams[i]);
            }
            
            cuCtxDestroy(context);

            var deleteEnd = System.nanoTime();

            deleteTimes[rep] = (deleteEnd - deleteStart) / 1e9;
        }
        
        var endWholeTime = System.nanoTime();

        logTimings(prepTimes, kernelTimes, copyTimes, deleteTimes, endWholeTime - startWholeTime);
    }

    private void logTimings(double[] prep, double[] kernel, double[] copy, double[] del, double wholeTime) {
        if (logReps) {
            for (var i = 0; i < prep.length; i++) {
                System.out.printf("  Repetition %d:\n", i + 1);
                System.out.printf("  Preparation time: %.6f s\n", prep[i]);
                System.out.printf("  Kernel execution time: %.6f s\n", kernel[i]);
                System.out.printf("  Data copy time: %.6f s\n", copy[i]);
                System.out.printf("  Memory deletion time: %.6f s\n", del[i]);
            }
        }

        var n = prep.length;
        var prepAvg = mean(prep);
        var kernelAvg = mean(kernel);
        var copyAvg = mean(copy);
        var delAvg = mean(del);
        var prepStd = stddev(prep, prepAvg);
        var kernelStd = stddev(kernel, kernelAvg);
        var copyStd = stddev(copy, copyAvg);
        var delStd = stddev(del, delAvg);

        System.out.printf("\nAverages over %d repetitions:\n", n);
        System.out.printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prepAvg, prepStd);
        System.out.printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernelAvg, kernelStd);
        System.out.printf("  Avg data copy time: %.6f s (stddev: %.6f s)\n", copyAvg, copyStd);
        System.out.printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delAvg, delStd);
        System.out.printf("  Whole time taken for %d reps: %.6f s\n",NUM_REPS, wholeTime / 1e9);
        System.out.println("=========================");
    }

    private double mean(double[] arr) {
        var sum = 0.0;
        for (var v : arr) sum += v;
        return sum / arr.length;
    }

    private double stddev(double[] arr, double mean) {
        var sum = 0.0;
        for (var v : arr) sum += (v - mean) * (v - mean);
        return Math.sqrt(sum / arr.length);
    }
    
    private void setConstant(CUmodule module, String name, float[] values) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuModuleGetGlobal(ptr, new long[1], module, name);
        cuMemcpyHtoD(ptr, Pointer.to(values), (long) values.length * Sizeof.FLOAT);
    }
    
    private void performColdRun() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        var device = new CUdevice();
        cuDeviceGet(device, 0);

        var context = new CUcontext();
        cuCtxCreate(context, 0, device);

        var chunkSize = LENGTH / NUM_STREAMS;
        var module = new CUmodule();
        cuModuleLoad(module, PTX_FILENAME);

        var function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        var streams = new CUstream[NUM_STREAMS];
        var deviceChunks = new CUdeviceptr[NUM_STREAMS];
        var hostChunkPtrs = new Pointer[NUM_STREAMS];
        var hostChunkBuffers = new java.nio.FloatBuffer[NUM_STREAMS];
        for (var i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], 0);

            var startIdx = i * chunkSize;
            var endIdx = Math.min(startIdx + chunkSize, LENGTH);
            var currentChunkSize = endIdx - startIdx;
            deviceChunks[i] = new CUdeviceptr();
            cuMemAlloc(deviceChunks[i], (long) currentChunkSize * Sizeof.FLOAT);
            hostChunkPtrs[i] = new Pointer();
            cuMemHostAlloc(hostChunkPtrs[i], (long) currentChunkSize * Sizeof.FLOAT, 0);
            hostChunkBuffers[i] = hostChunkPtrs[i].getByteBuffer(0, (long) currentChunkSize * Sizeof.FLOAT).asFloatBuffer();
        }
        for (var i = 0; i < NUM_STREAMS; i++) {
            var startIdx = i * chunkSize;
            var endIdx = Math.min(startIdx + chunkSize, LENGTH);
            var currentChunkSize = endIdx - startIdx;
            
            setConstant(module, "d_params", new float[]{
                TMIN + startIdx * DELTA,  // tmin
                TMAX,                     // tmax
                LENGTH,                   // length
                COEFFICIENTS,             // coefficients
                DELTA                     // delta
            });
            
            var coefficients = new float[COEFFICIENTS];
            for (int k = 0; k < COEFFICIENTS; k++) {
                coefficients[k] = 1.0f / (4.0f * (k + 1) * (k + 1) - 4.0f * (k + 1) + 1.0f);
            }
            setConstant(module, "d_coefficients", coefficients);
            
            var kernelParams = Pointer.to(
                Pointer.to(new int[]{startIdx}),
                Pointer.to(new int[]{endIdx}),
                Pointer.to(deviceChunks[i])
            );

            var blocks = (currentChunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            var sharedMemSize = COEFFICIENTS * Sizeof.FLOAT;
            cuLaunchKernel(function,
                blocks, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                sharedMemSize, streams[i],
                kernelParams, null);
            cuMemcpyDtoHAsync(hostChunkPtrs[i], deviceChunks[i], (long) currentChunkSize * Sizeof.FLOAT, streams[i]);
        }
        for (var stream : streams) {
            cuStreamSynchronize(stream);
        }
        for (var i = 0; i < NUM_STREAMS; i++) {
            cuMemFree(deviceChunks[i]);
            cuMemFreeHost(hostChunkPtrs[i]);
            cuStreamDestroy(streams[i]);
        }
        cuCtxDestroy(context);
    }
}
