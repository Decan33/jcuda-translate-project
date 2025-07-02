package org.jcuda.kacygan.mastersdeg;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
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
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventDestroy;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.CUevent;
import jcuda.driver.JCudaDriver;
import org.apache.commons.lang3.time.StopWatch;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;

@SuppressWarnings("java:S106")
public class FourierCalculator3 {
    public static final double THOUSAND = 1000.0;
    //Fourier: Streams + shared version
    private static final String PTX_FILENAME = "Fourier2.ptx";
    private static final String FUNCTION_NAME = "fourier";
    private static final int NUM_REPS = 20;
    private static final int NUM_STREAMS = 4;
    private static final int LENGTH = 1_000_000_000;
    private static final int COEFFICIENTS = 1024;
    private static final float TMIN = -3.0f;
    private static final float TMAX = 3.0f;
    private static final int THREADS_PER_BLOCK = 256;

    public static void main(String[] args) {
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
            var chunkSize = LENGTH / NUM_STREAMS;
            var module = new CUmodule();
            cuModuleLoad(module, PTX_FILENAME);

            var function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            setConstant(module, "const_delta", delta);
            setConstant(module, "const_coefficients", COEFFICIENTS);
            setConstant(module, "const_pi", (float) Math.PI);
            setConstant(module, "const_pi_squared", (float) (Math.PI * Math.PI));
            setConstant(module, "const_T", 1.0f);
            setConstant(module, "const_pi_over_T", (float) (Math.PI));
            setConstant(module, "constant_result_coefficient", (4.0f) / ((float) (Math.PI * Math.PI)));

            var streams = new CUstream[NUM_STREAMS];
            var deviceChunks = new CUdeviceptr[NUM_STREAMS];
            var hostChunks = new float[NUM_STREAMS][chunkSize];
            for (var i = 0; i < NUM_STREAMS; i++) {
                streams[i] = new CUstream();
                cuStreamCreate(streams[i], 0);

                deviceChunks[i] = new CUdeviceptr();
                cuMemAlloc(deviceChunks[i], chunkSize * Sizeof.FLOAT);
            }

            var prepEnd = System.nanoTime();
            prepTimes[rep] = (prepEnd - prepStart) / 1e9;

            var kernelStart = new CUevent();
            var kernelStop = new CUevent();
            cuEventCreate(kernelStart, 0);
            cuEventCreate(kernelStop, 0);
            cuEventRecord(kernelStart, null);

            for (var i = 0; i < NUM_STREAMS; i++) {
                var offset = i * chunkSize;
                var tminChunk = TMIN + offset * delta;
                setConstant(module, "const_tmin", tminChunk);
                setConstant(module, "const_chunk_size", chunkSize);
                var kernelParams = Pointer.to(Pointer.to(deviceChunks[i]));

                var blocks = (chunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                cuLaunchKernel(function,
                    blocks, 1, 1,
                    THREADS_PER_BLOCK, 1, 1,
                    0, streams[i],
                    kernelParams, null);
                cuMemcpyDtoHAsync(Pointer.to(hostChunks[i]), deviceChunks[i],
                    chunkSize * Sizeof.FLOAT, streams[i]);
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
            var deleteStart = System.nanoTime();

            for (var i = 0; i < NUM_STREAMS; i++) {
                cuMemFree(deviceChunks[i]);
                cuStreamDestroy(streams[i]);
            }
            cuCtxDestroy(context);

            var deleteEnd = System.nanoTime();

            deleteTimes[rep] = (deleteEnd - deleteStart) / 1e9;
        }
        var endWholeTime = System.nanoTime();

        logTimings(prepTimes, kernelTimes, deleteTimes, endWholeTime - startWholeTime);
    }

    private static void logTimings(double[] prep, double[] kernel, double[] del, double wholeTime) {
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
        var prepStd = stddev(prep, prepAvg);
        var kernelStd = stddev(kernel, kernelAvg);
        var delStd = stddev(del, delAvg);

        System.out.printf("\nAverages over %d repetitions:\n", n);
        System.out.printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prepAvg, prepStd);
        System.out.printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernelAvg, kernelStd);
        System.out.printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delAvg, delStd);
        System.out.printf("  Whole time taken for %d reps: %.6f s\n",NUM_REPS, wholeTime / 1e9);
        System.out.println("=========================");
    }

    private static double mean(double[] arr) {
        var sum = 0.0;
        for (var v : arr) sum += v;
        return sum / arr.length;
    }

    private static double stddev(double[] arr, double mean) {
        var sum = 0.0;
        for (var v : arr) sum += (v - mean) * (v - mean);
        return Math.sqrt(sum / arr.length);
    }

    private static void setConstant(CUmodule module, String name, float value) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuModuleGetGlobal(ptr, new long[1], module, name);
        cuMemcpyHtoD(ptr, Pointer.to(new float[]{value}), Sizeof.FLOAT);
    }

    private static void setConstant(CUmodule module, String name, int value) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuModuleGetGlobal(ptr, new long[1], module, name);
        cuMemcpyHtoD(ptr, Pointer.to(new int[]{value}), Sizeof.INT);
    }

    private static void writeResultsToCSV(String filename, float tmin, float delta, float[] results) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("t,f");
            for (int i = 0; i < results.length; i++) {
                float t = tmin + i * delta;
                writer.printf(Locale.US, "%.6f,%.6f%n", t, results[i]);
            }
        } catch (IOException e) {
            System.err.println("Error writing CSV: " + e.getMessage());
        }
    }
}
