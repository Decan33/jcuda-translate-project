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

@SuppressWarnings("java:S106")
public class FourierCalculator2 {

    public static final String FUNCTION_NAME = "fourier";
    public static final String KERNEL_PTX_FILENAME = "Fourier2.ptx";

    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        final float tmin = -3.0f;
        final float tmax = 3.0f;
        final int length = 256;
        final int coefficients = 1024;

        final float delta = (tmax - tmin) / (length - 1);
        final float pi = 3.14159265f;
        final float piSquared = pi * pi;
        final float period = 1.0f;
        final float piOverT = pi / period;
        final float resultCoefficient = (4.0f * period) / piSquared;

        CUdeviceptr dResults = new CUdeviceptr();
        cuMemAlloc(dResults, length * Sizeof.FLOAT);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, KERNEL_PTX_FILENAME);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        setAllConstantsInGpuMemory(tmin, coefficients, delta, pi, piSquared, period, piOverT, resultCoefficient, module);

        final int threadsPerBlock = 256;
        final int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;

        Pointer kernelParameters = Pointer.to(Pointer.to(dResults));
        cuLaunchKernel(function, blocks, 1,
                1, threadsPerBlock, 1,
                1, 0, null,
                kernelParameters, null);
        cuCtxSynchronize();

        float[] hostResults = new float[length];
        cuMemcpyDtoH(Pointer.to(hostResults), dResults, length * Sizeof.FLOAT);

        cuMemFree(dResults);
        cuCtxDestroy(context);

        writeResultsToCSV("results_" + coefficients + "coeffs_extended.csv", tmin, delta, hostResults);

        System.out.println("Computation and CSV export done successfully.");
    }

    private static void setAllConstantsInGpuMemory(float tmin, int coefficients, float delta, float pi, float piSquared, float period, float piOverT, float resultCoefficient, CUmodule module) {
        setConstant(module, "const_tmin", tmin);
        setConstant(module, "const_delta", delta);
        setConstant(module, "const_coefficients", coefficients);
        setConstant(module, "const_pi", pi);
        setConstant(module, "const_pi_squared", piSquared);
        setConstant(module, "const_T", period);
        setConstant(module, "const_pi_over_T", piOverT);
        setConstant(module, "constant_result_coefficient", resultCoefficient);
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
