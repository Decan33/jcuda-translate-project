package org.jcuda.kacygan.mastersdeg;

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
import static jcuda.driver.JCudaDriver.cuStreamCreate;
import static jcuda.driver.JCudaDriver.cuStreamDestroy;
import static jcuda.driver.JCudaDriver.cuStreamSynchronize;

@SuppressWarnings("java:S106")
public class FourierCalculator4 {
    //Fourier: streams version
    static StopWatch watch = new StopWatch();
    public static final String PTX_FILENAME = "Fourier.ptx";
    public static final String FUNCTION_NAME = "fourier";
    private static final int NUM_STREAMS = 4;
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


//        int length = 200000000;
//        int length = 500000000;
            int length = 1000000000;
//        int length = 2000000000;

            int coefficients = 1024;
            float delta = (tmax - tmin) / (length - 1);

            CUstream[] streams = new CUstream[NUM_STREAMS];
            for (int i = 0; i < NUM_STREAMS; i++) {
                streams[i] = new CUstream();
                cuStreamCreate(streams[i], 0);
            }

            int chunkSize = (length + NUM_STREAMS - 1) / NUM_STREAMS;
            CUdeviceptr[] deviceResults = new CUdeviceptr[NUM_STREAMS];
            float[][] hostResults = new float[NUM_STREAMS][];

            CUmodule module = new CUmodule();
            cuModuleLoad(module, PTX_FILENAME);

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, FUNCTION_NAME);

            int threadsPerBlock = 256;

            for (int i = 0; i < NUM_STREAMS; i++) {
                int startIdx = i * chunkSize;
                int currentChunkSize = Math.min(chunkSize, length - startIdx);

                deviceResults[i] = new CUdeviceptr();
                cuMemAlloc(deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
                hostResults[i] = new float[currentChunkSize];

                float chunkTmin = tmin + startIdx * delta;

                Pointer kernelParameters = Pointer.to(
                        Pointer.to(new float[]{chunkTmin}),
                        Pointer.to(new float[]{delta}),
                        Pointer.to(new int[]{currentChunkSize}),
                        Pointer.to(new int[]{coefficients}),
                        Pointer.to(deviceResults[i])
                );

                int blocksPerGrid = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;

                cuLaunchKernel(function,
                        blocksPerGrid, 1, 1,
                        threadsPerBlock, 1, 1,
                        0, streams[i],
                        kernelParameters, null
                );

                cuMemcpyDtoH(Pointer.to(hostResults[i]), deviceResults[i], (long)currentChunkSize * Sizeof.FLOAT);
            }

            for (CUstream stream : streams) {
                cuStreamSynchronize(stream);
            }

            float[] finalResults = new float[length];
            for (int i = 0; i < NUM_STREAMS; i++) {
                int startIdx = i * chunkSize;
                int currentChunkSize = Math.min(chunkSize, length - startIdx);
                System.arraycopy(hostResults[i], 0, finalResults, startIdx, currentChunkSize);
            }

            for (int i = 0; i < NUM_STREAMS; i++) {
                cuMemFree(deviceResults[i]);
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
}
