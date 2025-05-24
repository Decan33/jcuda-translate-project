package org.jcuda.kacygan.mastersdeg;

import static jcuda.driver.JCudaDriver.*;
import jcuda.*;
import jcuda.driver.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class FourierCalculator3 {
    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Parameters
        final int NUM_STREAMS = 4;
        final int threadsPerBlock = 256;
        final int length = 1024 * 64;
        final int coefficients = 1024;

        float tmin = -3.0f;
        float tmax = 3.0f;
        float delta = (tmax - tmin) / (length - 1);
        int chunkSize = length / NUM_STREAMS;

        float pi = 3.14159265f;
        float piSq = pi * pi;
        float T = 1.0f;
        float piOverT = pi / T;
        float resultCoeff = (4.0f * T) / piSq;

        // Load PTX
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "Fourier2.ptx");

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "fourier");

        // Set constant memory (shared for all streams)
        setConstant(module, "const_delta", delta);
        setConstant(module, "const_coefficients", coefficients);
        setConstant(module, "const_pi", pi);
        setConstant(module, "const_pi_squared", piSq);
        setConstant(module, "const_T", T);
        setConstant(module, "const_pi_over_T", piOverT);
        setConstant(module, "constant_result_coefficient", resultCoeff);

        // Allocate arrays
        CUstream[] streams = new CUstream[NUM_STREAMS];
        CUdeviceptr[] deviceChunks = new CUdeviceptr[NUM_STREAMS];
        float[][] hostChunks = new float[NUM_STREAMS][chunkSize];

        for (int i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], 0);

            deviceChunks[i] = new CUdeviceptr();
            cuMemAlloc(deviceChunks[i], chunkSize * Sizeof.FLOAT);
        }

        // Launch per stream
        for (int i = 0; i < NUM_STREAMS; i++) {
            int offset = i * chunkSize;
            float tminChunk = tmin + offset * delta;

            // Update const_tmin for each chunk
            setConstant(module, "const_tmin", tminChunk);

            Pointer kernelParams = Pointer.to(Pointer.to(deviceChunks[i]));
            int blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

            cuLaunchKernel(function,
                    blocks, 1, 1,
                    threadsPerBlock, 1, 1,
                    0, streams[i],
                    kernelParams, null);

            cuMemcpyDtoHAsync(Pointer.to(hostChunks[i]), deviceChunks[i],
                    chunkSize * Sizeof.FLOAT, streams[i]);
        }

        // Wait for all streams
        for (CUstream stream : streams) {
            cuStreamSynchronize(stream);
        }

        // Combine results
        float[] hostResults = new float[length];
        for (int i = 0; i < NUM_STREAMS; i++) {
            System.arraycopy(hostChunks[i], 0, hostResults, i * chunkSize, chunkSize);
        }

        // Output to CSV
        writeResultsToCSV("result_stream_"+ (length > 300 ? "FIRST" : length) + ".csv", tmin, delta, hostResults);

        // Cleanup
        for (int i = 0; i < NUM_STREAMS; i++) {
            cuMemFree(deviceChunks[i]);
            cuStreamDestroy(streams[i]);
        }

        cuCtxDestroy(context);
        System.out.println("Done with streams + async + CSV export.");
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
