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

import static jcuda.driver.JCudaDriver.*;

@SuppressWarnings("java:S106")
public class FourierCalculator implements FourierTest {
    private static final String KERNEL_PTX_FILENAME = "Fourier.ptx";

    private void logTimings(double[] prep, double[] kernel, double[] del, double wholeTime) {
        if (logReps) {
            for (var i = 0; i < prep.length; i++) {
                System.out.printf("  Repetition %d:\n", i + 1);
                System.out.printf("  Preparation time: %.6f s\n", prep[i]);
                System.out.printf("  Kernel execution time: %.6f s\n", kernel[i]);
                System.out.printf("  Memory deletion time: %.6f s\n", del[i]);
            }
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
        System.out.printf("  Whole time taken for %d reps: %.6f s\n",NUM_REPS, wholeTime / 1e9);
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

        var deviceResults = new CUdeviceptr();
        cuMemAlloc(deviceResults, (long)LENGTH * Sizeof.FLOAT);

        var module = new CUmodule();
        cuModuleLoad(module, KERNEL_PTX_FILENAME);

        var function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        var kernelParameters = Pointer.to(
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

        var blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        cuLaunchKernel(function,
            blocksPerGrid, 1, 1,
            THREADS_PER_BLOCK, 1, 1,
            0, null,
            kernelParameters, null
        );

        var hostResults = new float[LENGTH];
        cuMemcpyDtoH(Pointer.to(hostResults), deviceResults, (long)LENGTH * Sizeof.FLOAT);
        
        cuMemFree(deviceResults);
        cuCtxDestroy(context);
    }

    @Override
    public void runTest() {
        System.out.println("TESTING RAW FOURIER");

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

            var deviceResults = new CUdeviceptr();
            cuMemAlloc(deviceResults, (long)LENGTH * Sizeof.FLOAT);

            var module = new CUmodule();
            cuModuleLoad(module, KERNEL_PTX_FILENAME);

            var function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            var kernelParameters = Pointer.to(
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

            var blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            var prepEnd = System.nanoTime();
            prepTimes[rep] = (prepEnd - prepStart) / 1e9;

            var kernelStart = new CUevent();
            var kernelStop = new CUevent();

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

            var kernelMs = new float[1];
            cuEventElapsedTime(kernelMs, kernelStart, kernelStop);

            kernelTimes[rep] = kernelMs[0] / THOUSAND;

            cuEventDestroy(kernelStart);
            cuEventDestroy(kernelStop);

            var deleteStart = System.nanoTime();
            var hostResults = new float[LENGTH];

            cuMemcpyDtoH(Pointer.to(hostResults), deviceResults, (long)LENGTH * Sizeof.FLOAT);
            cuMemFree(deviceResults);
            cuCtxDestroy(context);

            var deleteEnd = System.nanoTime();
            deleteTimes[rep] = (deleteEnd - deleteStart) / 1e9;
        }

        var endWholeTime = System.nanoTime();

        logTimings(prepTimes, kernelTimes, deleteTimes, endWholeTime - startWholeTime);
    }
}
