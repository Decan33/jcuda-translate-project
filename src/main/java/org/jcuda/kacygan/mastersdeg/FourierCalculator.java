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

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Locale;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;

public class FourierCalculator {
    public static void main(String[] args) throws Exception {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Kernel parameters
        float tmin = -3.0f;
        float tmax = 3.0f;
        int length = 65536 * 32;
        int coefficients = 1024;
        float delta = (tmax - tmin) / (length - 1);

        CUdeviceptr deviceResults = new CUdeviceptr();
        cuMemAlloc(deviceResults, (long)length * Sizeof.FLOAT);

        String ptxFileName = "Fourier.ptx";
        String ptxSource = new String(Files.readAllBytes(Paths.get(ptxFileName)));

        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptxSource);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "fourier");

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new float[]{tmin}),
                Pointer.to(new float[]{delta}),
                Pointer.to(new int[]{length}),
                Pointer.to(new int[]{coefficients}),
                Pointer.to(deviceResults)
        );

        int threadsPerBlock = 256;
        int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

        CUevent start = new CUevent();
        CUevent stop = new CUevent();
        cuEventCreate(start, 0);
        cuEventCreate(stop, 0);
        cuEventRecord(start, null);

        cuLaunchKernel(function,
                blocksPerGrid, 1, 1,
                threadsPerBlock, 1, 1,
                0, null,
                kernelParameters, null
        );

        cuEventRecord(stop, null);
        cuEventSynchronize(stop);

        float[] milliseconds = new float[1];
        cuEventElapsedTime(milliseconds, start, stop);
        System.out.printf("Kernel execution time: %3.1f ms%n", milliseconds[0]);

        // Copy results from GPU to host
        float[] hostResults = new float[length];
        cuMemcpyDtoH(Pointer.to(hostResults), deviceResults, (long)length * Sizeof.FLOAT);

        String csvFile = "results_"+ coefficients + "coeffs.csv";
        try (PrintWriter writer = new PrintWriter(new FileWriter(csvFile))) {
            // Optionally write header:
            writer.println("t,f");

            for (int i = 0; i < hostResults.length; i++) {
                var t = tmin + i * delta;

                writer.printf(Locale.US, "%f,%.6f%n", t, hostResults[i]);
            }

            System.out.println("Results successfully exported to " + csvFile);
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
        }

        // Cleanup resources
        cuMemFree(deviceResults);
        cuCtxDestroy(context);
    }
}
