package org.jcuda.kacygan.mastersdeg;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import org.apache.commons.lang3.time.StopWatch;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

@SuppressWarnings("java:S106")
public class FourierCalculator {
    static StopWatch watch = new StopWatch();
    public static final String FUNCTION_NAME = "fourier";
    public static final String KERNEL_PTX_FILENAME = "Fourier.ptx";
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

            float tmin = -3.0f;
            float tmax = 3.0f;

//            int length = 200000000;
//        int length = 500000000;
//        int length = 1000000000;
        int length = 2000000000;
            int coefficients = 1024;
            float delta = (tmax - tmin) / (length - 1);

            CUdeviceptr deviceResults = new CUdeviceptr();
            cuMemAlloc(deviceResults, (long)length * Sizeof.FLOAT);

            CUmodule module = new CUmodule();
            cuModuleLoad(module, KERNEL_PTX_FILENAME);
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(new float[]{tmin}),
                    Pointer.to(new float[]{delta}),
                    Pointer.to(new int[]{length}),
                    Pointer.to(new int[]{coefficients}),
                    Pointer.to(deviceResults)
            );

            int threadsPerBlock = 256;
            int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

//        CUevent start = new CUevent();
//        CUevent stop = new CUevent();
//        cuEventCreate(start, 0);
//        cuEventCreate(stop, 0);
//        cuEventRecord(start, null);

            cuLaunchKernel(function,
                    blocksPerGrid, 1, 1,
                    threadsPerBlock, 1, 1,
                    0, null,
                    kernelParameters, null
            );

//        cuEventRecord(stop, null);
//        cuEventSynchronize(stop);

//        float[] milliseconds = new float[1];
//        cuEventElapsedTime(milliseconds, start, stop);
//        System.out.printf("Kernel execution time: %3.1f ms%n", milliseconds[0]);

            float[] hostResults = new float[length];
            cuMemcpyDtoH(Pointer.to(hostResults), deviceResults, (long)length * Sizeof.FLOAT);

//        String csvFile = "results_"+ coefficients + "coeffs.csv";
//        try (PrintWriter writer = new PrintWriter(new FileWriter(csvFile))) {
//            writer.println("t,f");
//
//            for (int i = 0; i < hostResults.length; i++) {
//                var t = tmin + i * delta;
//
//                writer.printf(Locale.US, "%f,%.6f%n", t, hostResults[i]);
//            }
//
//            System.out.println("Results successfully exported to " + csvFile);
//        } catch (IOException e) {
//            System.err.println("Error writing to CSV file: " + e.getMessage());
//        }

            cuMemFree(deviceResults);
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

        System.out.printf("Time took for all that: %.4f\n", accum / 10.0);
        System.out.printf("Standard deviation: %f\n", stdev);
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
}
