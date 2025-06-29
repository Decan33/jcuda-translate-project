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

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.apache.commons.lang3.time.StopWatch;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;

@SuppressWarnings("java:S106")
public class FourierCalculator3 {
    //Fourier: Streams + shared version
    static StopWatch watch = new StopWatch();

    public static final String PTX_FILENAME = "Fourier2.ptx";
    public static final String FUNCTION_NAME = "fourier";
    public static int NUM_REPS = 1;


    static double run() {
        watch.start();
        for (int rep = 0; rep < NUM_REPS; rep++) {
            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);

            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);

            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);

            final int NUM_STREAMS = 4;
            final int threadsPerBlock = 256;

//            int length = 200000000;
//        int length = 500000000;
//        int length = 1000000000;
        int length = 2000000000;

            final var coefficients = 1024;

            final var minimum = -3.0f;
            final var maximum = 3.0f;
            final var delta = (maximum - minimum) / (length - 1);
            final var chunkSize = length / NUM_STREAMS;

            final var piSq = (float) (Math.PI * Math.PI);
            final var period = 1.0f;
            final var piOverT = (float) (Math.PI / period);
            final var resultCoefficient = (4.0f * period) / piSq;

            var module = new CUmodule();
            cuModuleLoad(module, PTX_FILENAME);

            var function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            setConstant(module, "const_delta", delta);
            setConstant(module, "const_coefficients", coefficients);
            setConstant(module, "const_pi", (float) Math.PI);
            setConstant(module, "const_pi_squared", piSq);
            setConstant(module, "const_T", period);
            setConstant(module, "const_pi_over_T", piOverT);
            setConstant(module, "constant_result_coefficient", resultCoefficient);

            var streams = new CUstream[NUM_STREAMS];
            var deviceChunks = new CUdeviceptr[NUM_STREAMS];
            var hostChunks = new float[NUM_STREAMS][chunkSize];

            for (int i = 0; i < NUM_STREAMS; i++) {
                streams[i] = new CUstream();
                cuStreamCreate(streams[i], 0);

                deviceChunks[i] = new CUdeviceptr();
                cuMemAlloc(deviceChunks[i], chunkSize * Sizeof.FLOAT);
            }

            for (int i = 0; i < NUM_STREAMS; i++) {
                var offset = i * chunkSize;
                var tminChunk = minimum + offset * delta;

                setConstant(module, "const_tmin", tminChunk);

                var kernelParams = Pointer.to(Pointer.to(deviceChunks[i]));
                var blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

                cuLaunchKernel(function,
                        blocks, 1, 1,
                        threadsPerBlock, 1, 1,
                        0, streams[i],
                        kernelParams, null);

                cuMemcpyDtoHAsync(Pointer.to(hostChunks[i]), deviceChunks[i],
                        chunkSize * Sizeof.FLOAT, streams[i]);
            }

            for (CUstream stream : streams) {
                cuStreamSynchronize(stream);
            }

//        float[] hostResults = new float[length];
//        for (int i = 0; i < NUM_STREAMS; i++) {
//            System.arraycopy(hostChunks[i], 0, hostResults, i * chunkSize, chunkSize);
//        }

//        writeResultsToCSV("result_stream_"+ (length > 300 ? "FIRST" : length) + ".csv", tmin, delta, hostResults);

            for (int i = 0; i < NUM_STREAMS; i++) {
                cuMemFree(deviceChunks[i]);
                cuStreamDestroy(streams[i]);
            }

            cuCtxDestroy(context);
        }

        watch.stop();
        return watch.getTime() / 1000.0;
    }

    public static void main(String[] args) {
        for (int i = 0; i < 5; i++) {
            watch = new StopWatch();

            run();
        }

        NUM_REPS = 5;
        double accum = 0.0;
        var nums = new double[10];
        for (int i = 0; i < 10; i++) {
            watch = new StopWatch();
            nums[i] = run();

            accum += nums[i];
        }

        var mean = accum / 10.0;
        var stdev = calculateSD(nums, mean);

        System.out.printf("Time took for all that: %.4f", accum / 10.0);
        System.out.printf("Standard deviation: %f", stdev);
    }

    public static double calculateSD(double numArray[], double mean)
    {
        double standardDeviation = 0.0;
        int length = numArray.length;

        for(double num: numArray) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        return Math.sqrt(standardDeviation/length);
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
