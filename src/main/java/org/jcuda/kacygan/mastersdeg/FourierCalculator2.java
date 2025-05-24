package org.jcuda.kacygan.mastersdeg;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
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

public class FourierCalculator2 {
    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        float tmin = -3.0f;
        float tmax = 3.0f;
        int length = 256;
        int coefficients = 1024;

        float delta = (tmax - tmin) / (length - 1);
        float pi = 3.14159265f;
        float piSquared = pi * pi;
        float T = 1.0f;
        float piOverT = pi / T;
        float resultCoefficient = (4.0f * T) / piSquared;

        // Device memory allocation
        CUdeviceptr dResults = new CUdeviceptr();
        cuMemAlloc(dResults, length * Sizeof.FLOAT);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, "Fourier2.ptx");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "fourier");

        setConstant(module, "const_tmin", tmin);
        setConstant(module, "const_delta", delta);
        setConstant(module, "const_coefficients", coefficients);
        setConstant(module, "const_pi", pi);
        setConstant(module, "const_pi_squared", piSquared);
        setConstant(module, "const_T", T);
        setConstant(module, "const_pi_over_T", piOverT);
        setConstant(module, "constant_result_coefficient", resultCoefficient);

        int threadsPerBlock = 256;
        int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;

        Pointer kernelParameters = Pointer.to(Pointer.to(dResults));
        cuLaunchKernel(function, blocks, 1,
                1, threadsPerBlock, 1,
                1, 0, null,
                kernelParameters, null);
        cuCtxSynchronize();

        float[] hostResults = new float[length];
        cuMemcpyDtoH(Pointer.to(hostResults), dResults, length * Sizeof.FLOAT);

//        writeResultsToCSV("result_256.csv", tmin, delta, hostResults);

        cuMemFree(dResults);
        cuCtxDestroy(context);

        System.out.println("Computation and CSV export done successfully.");
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
