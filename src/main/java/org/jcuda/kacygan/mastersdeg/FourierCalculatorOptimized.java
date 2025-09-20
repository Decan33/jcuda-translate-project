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
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleGetGlobal;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;
import static jcuda.driver.JCudaDriver.cuEventDestroy;

@SuppressWarnings("java:S106")
public class FourierCalculatorOptimized implements FourierTest {
    private static final String KERNEL_PTX_FILENAME = "FourierOptimized.ptx";

    @Override
    public void runTest() {
        System.out.println("TESTING FOURIER WITH CONSTANTS AND SHARED MEMORY");

        System.out.println("Performing cold run to warm up GPU...");
        performColdRun();
        System.out.println("Cold run completed.\n");

        var prepTimes = new double[NUM_REPS];
        var kernelTimes = new double[NUM_REPS];
        var copyTimes = new double[NUM_REPS];
        var deleteTimes = new double[NUM_REPS];

        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        var device = new CUdevice();
        cuDeviceGet(device, 0);

        var context = new CUcontext();
        cuCtxCreate(context, 0, device);

        var startWholeTime = System.nanoTime();
        
        for (var rep = 0; rep < NUM_REPS; rep++) {

            var prepStart = System.nanoTime();

            var deviceResults = new CUdeviceptr();
            cuMemAlloc(deviceResults, (long) LENGTH * Sizeof.FLOAT);

            var module = new CUmodule();
            cuModuleLoad(module, KERNEL_PTX_FILENAME);

            var function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            setAllConstants(module);

            var kernelParameters = Pointer.to(Pointer.to(deviceResults));
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

            var copyStart = System.nanoTime();
            var hostResults = new float[LENGTH];
            cuMemcpyDtoH(Pointer.to(hostResults), deviceResults, (long) LENGTH * Sizeof.FLOAT);
            var copyEnd = System.nanoTime();
            copyTimes[rep] = (copyEnd - copyStart) / 1e9;

            var deleteStart = System.nanoTime();
            cuMemFree(deviceResults);
            var deleteEnd = System.nanoTime();
            deleteTimes[rep] = (deleteEnd - deleteStart) / 1e9;
        }
        
        var endWholeTime = System.nanoTime();

        cuCtxDestroy(context);

        logTimings(prepTimes, kernelTimes, copyTimes, deleteTimes, endWholeTime - startWholeTime);
    }

    private void setAllConstants(CUmodule module) {
        setConstant(module, "const_tmin", TMIN);
        setConstant(module, "const_delta", DELTA);
        setConstant(module, "const_coefficients", COEFFICIENTS);
        setConstant(module, "const_pi", PI);
        setConstant(module, "const_pi_squared", PI_SQUARED);
        setConstant(module, "const_T", PERIOD);
        setConstant(module, "const_pi_over_T", PI_OVER_T);
        setConstant(module, "constant_result_coefficient", RESULT_COEFFICIENT);
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
        var prepStd = standardDeviation(prep, prepAvg);
        var kernelStd = standardDeviation(kernel, kernelAvg);
        var copyStd = standardDeviation(copy, copyAvg);
        var delStd = standardDeviation(del, delAvg);

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

    private double standardDeviation(double[] arr, double mean) {
        var sum = 0.0;
        for (var v : arr) sum += (v - mean) * (v - mean);
        return Math.sqrt(sum / arr.length);
    }

    private void setConstant(CUmodule module, String name, float value) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuModuleGetGlobal(ptr, new long[1], module, name);
        cuMemcpyHtoD(ptr, Pointer.to(new float[]{value}), Sizeof.FLOAT);
    }

    private void setConstant(CUmodule module, String name, int value) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuModuleGetGlobal(ptr, new long[1], module, name);
        cuMemcpyHtoD(ptr, Pointer.to(new int[]{value}), Sizeof.INT);
    }

    private void performColdRun() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        var device = new CUdevice();
        cuDeviceGet(device, 0);

        var context = new CUcontext();
        cuCtxCreate(context, 0, device);

        var deviceResults = new CUdeviceptr();
        cuMemAlloc(deviceResults, (long) LENGTH * Sizeof.FLOAT);

        var module = new CUmodule();
        cuModuleLoad(module, KERNEL_PTX_FILENAME);

        var function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        setAllConstants(module);

        var kernelParameters = Pointer.to(Pointer.to(deviceResults));
        var blocksPerGrid = (LENGTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                THREADS_PER_BLOCK, 1, 1,
                0, null,
                kernelParameters, null
        );

        var hostResults = new float[LENGTH];
        cuMemcpyDtoH(Pointer.to(hostResults), deviceResults, (long) LENGTH * Sizeof.FLOAT);
        
        cuMemFree(deviceResults);
        cuCtxDestroy(context);
    }
}
