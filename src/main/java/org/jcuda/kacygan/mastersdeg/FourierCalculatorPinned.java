package org.jcuda.kacygan.mastersdeg;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
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
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.runtime.JCuda.cudaFreeHost;
import static jcuda.runtime.JCuda.cudaHostAlloc;
import static jcuda.runtime.JCuda.cudaHostAllocDefault;

@SuppressWarnings("java:S106")
public class FourierCalculatorPinned implements FourierTest{
    private static final String KERNEL_PTX_FILENAME = "FourierRaw.ptx";
    private static final int CHUNK_SIZE = 250_000_000;

    @Override
    public void runTest() {
        try {
            long requiredDeviceMemory = (long)LENGTH * Sizeof.FLOAT;
            long requiredHostMemory = (long)Math.min(CHUNK_SIZE, LENGTH) * Sizeof.FLOAT;
            
            System.out.printf("Memory requirements: Device=%.2f GB, Host chunk=%.2f MB\n", 
                             requiredDeviceMemory / (1024.0 * 1024.0 * 1024.0),
                             requiredHostMemory / (1024.0 * 1024.0));

            System.out.println("TESTING FOURIER USING PINNED MEMORY");

            System.out.println("Performing cold run to warm up GPU...");
            performColdRun();
            System.out.println("Cold run completed.\n");
            
            double[] prepTimes = new double[NUM_REPS];
            double[] kernelTimes = new double[NUM_REPS];
            double[] copyTimes = new double[NUM_REPS];
            double[] deleteTimes = new double[NUM_REPS];

            CUcontext context = new CUcontext();

            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            cuCtxCreate(context, 0, device);
            
            long startWholeTime = System.nanoTime();

            for (int rep = 0; rep < NUM_REPS; rep++) {
                long prepStart = System.nanoTime();
                
                JCudaDriver.setExceptionsEnabled(true);
                cuInit(0);

                CUdeviceptr deviceResults = new CUdeviceptr();
                cuMemAlloc(deviceResults, (long)LENGTH * Sizeof.FLOAT);
                
                CUmodule module = new CUmodule();
                cuModuleLoad(module, KERNEL_PTX_FILENAME);
                
                CUfunction function = new CUfunction();
                cuModuleGetFunction(function, module, FUNCTION_NAME);
                
                Pointer kernelParameters = Pointer.to(
                    Pointer.to(new float[]{TMIN}),
                    Pointer.to(new float[]{DELTA}),
                    Pointer.to(new int[]{LENGTH}),
                    Pointer.to(new int[]{COEFFICIENTS}),
                    Pointer.to(new float[]{PI}),
                    Pointer.to(new float[]{PI_OVER_T}),
                    Pointer.to(new float[]{RESULT_COEFFICIENT}),
                    Pointer.to(new float[]{PERIOD}),
                    Pointer.to(deviceResults)
                );
                
                int blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                
                long prepEnd = System.nanoTime();
                prepTimes[rep] = (prepEnd - prepStart) / 1e9;
                
                CUevent kernelStart = new CUevent();
                CUevent kernelStop = new CUevent();
                cuEventCreate(kernelStart, 0);
                cuEventCreate(kernelStop, 0);
                
                cuEventRecord(kernelStart, null);
                cuLaunchKernel(function,
                    blocksPerGrid, 1, 1,
                    THREADS_PER_BLOCK, 1, 1,
                    0, null,
                    kernelParameters, null
                );
                cuEventRecord(kernelStop, null);
                cuEventSynchronize(kernelStop);
                
                float[] kernelMs = new float[1];
                cuEventElapsedTime(kernelMs, kernelStart, kernelStop);
                kernelTimes[rep] = kernelMs[0] / 1000.0;
                
                cuEventDestroy(kernelStart);
                cuEventDestroy(kernelStop);
                
                long copyStart = System.nanoTime();
                int actualChunkSize = Math.min(CHUNK_SIZE, LENGTH);
                Pointer hostResults = new Pointer();
                cudaHostAlloc(hostResults, (long)actualChunkSize * Sizeof.FLOAT, cudaHostAllocDefault);
                for (int offset = 0; offset < LENGTH; offset += actualChunkSize) {
                    int thisChunk = Math.min(actualChunkSize, LENGTH - offset);
                    long byteOffset = (long)offset * Sizeof.FLOAT;
                    cuMemcpyDtoH(hostResults,
                                 deviceResults.withByteOffset(byteOffset),
                                 (long)thisChunk * Sizeof.FLOAT);
                }
                long copyEnd = System.nanoTime();
                copyTimes[rep] = (copyEnd - copyStart) / 1e9;

                long deleteStart = System.nanoTime();
                cudaFreeHost(hostResults);
                cuMemFree(deviceResults);
                long deleteEnd = System.nanoTime();
                deleteTimes[rep] = (deleteEnd - deleteStart) / 1e9;
            }
            
            long endWholeTime = System.nanoTime();

            cuCtxDestroy(context);

            logTimings(prepTimes, kernelTimes, copyTimes, deleteTimes, endWholeTime - startWholeTime);
            
        } catch (Exception e) {
            System.err.println("Error in FourierCalculatorPinned: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void logTimings(double[] prep, double[] kernel, double[] copy, double[] del, long wholeTime) {
        if (logReps) {
            for (var i = 0; i < prep.length; i++) {
                System.out.printf("  Repetition %d:\n", i + 1);
                System.out.printf("  Preparation time: %.6f s\n", prep[i]);
                System.out.printf("  Kernel execution time: %.6f s\n", kernel[i]);
                System.out.printf("  Data copy time: %.6f s\n", copy[i]);
                System.out.printf("  Memory deletion time: %.6f s\n", del[i]);
            }
        }

        int n = prep.length;
        double prepAvg = mean(prep);
        double kernelAvg = mean(kernel);
        double copyAvg = mean(copy);
        double delAvg = mean(del);
        double prepStd = standardDeviation(prep, prepAvg);
        double kernelStd = standardDeviation(kernel, kernelAvg);
        double copyStd = standardDeviation(copy, copyAvg);
        double delStd = standardDeviation(del, delAvg);

        System.out.printf("\nAverages over %d repetitions:\n", n);
        System.out.printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prepAvg, prepStd);
        System.out.printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernelAvg, kernelStd);
        System.out.printf("  Avg data copy time: %.6f s (stddev: %.6f s)\n", copyAvg, copyStd);
        System.out.printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delAvg, delStd);
        System.out.printf("  Whole time taken for %d reps: %.6f s\n", n, wholeTime / 1e9);
        System.out.println("=========================");
    }

    private static double mean(double[] arr) {
        double sum = 0.0;
        for (double v : arr) sum += v;
        return sum / arr.length;
    }

    private static double standardDeviation(double[] arr, double mean) {
        double sum = 0.0;
        for (double v : arr) sum += (v - mean) * (v - mean);
        return Math.sqrt(sum / arr.length);
    }
    
    private static void performColdRun() {
        try {
            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);
            
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);
            
            CUdeviceptr deviceResults = new CUdeviceptr();
            cuMemAlloc(deviceResults, (long)LENGTH * Sizeof.FLOAT);
            
            CUmodule module = new CUmodule();
            cuModuleLoad(module, KERNEL_PTX_FILENAME);
            
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);
            
            Pointer kernelParameters = Pointer.to(
                Pointer.to(new float[]{TMIN}),
                Pointer.to(new float[]{DELTA}),
                Pointer.to(new int[]{LENGTH}),
                Pointer.to(new int[]{COEFFICIENTS}),
                Pointer.to(new float[]{PI}),
                Pointer.to(new float[]{PI_OVER_T}),
                Pointer.to(new float[]{RESULT_COEFFICIENT}),
                Pointer.to(new float[]{PERIOD}),
                Pointer.to(deviceResults)
            );
            
            int blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            
            cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                0, null,
                kernelParameters, null
            );
            
            int actualChunkSize = Math.min(CHUNK_SIZE, LENGTH);
            Pointer hostResults = new Pointer();
            cudaHostAlloc(hostResults, (long)actualChunkSize * Sizeof.FLOAT, cudaHostAllocDefault);
            
            for (int offset = 0; offset < LENGTH; offset += actualChunkSize) {
                int thisChunk = Math.min(actualChunkSize, LENGTH - offset);
                long byteOffset = (long)offset * Sizeof.FLOAT;
                
                cuMemcpyDtoH(hostResults,
                             deviceResults.withByteOffset(byteOffset),
                             (long)thisChunk * Sizeof.FLOAT);
            }
            
            cudaFreeHost(hostResults);
            cuMemFree(deviceResults);
            cuCtxDestroy(context);
            
        } catch (Exception e) {
            System.err.println("Error in cold run: " + e.getMessage());
            e.printStackTrace();
        }
    }
}