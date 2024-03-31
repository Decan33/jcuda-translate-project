package org.jcuda.kacygan.chapter4;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import org.jcuda.kacygan.chapter3.JCudaCode;

import java.util.stream.IntStream;

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
import static jcuda.driver.JCudaDriver.cuModuleLoad;

public class AddLoopGpu extends JCudaCode {

    private static String KERNEL_SOURCE_CODE = """
            extern "C"
            __global__ void add( int *a, int *b, int *c ) {
                int tid = blockIdx.x;    // this thread handles the data at its thread id
                if (tid < N)
                    c[tid] = a[tid] + b[tid];
            }
            """;

    private static boolean COMPILATION_AS_STRING = true;
    private static final Integer N = 10;
    private static final long SIZE = N * Sizeof.INT;


    public static void main(String[] args) {
        // Initialize the driver and create a context for the first device
        cuInit(0);
        CUcontext context = new CUcontext();
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Load the PTX file
        CUmodule module = new CUmodule();
        cuModuleLoad(module, "VectorAdd.ptx");
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        var a = new int[N];
        var b = new int[N];
        var c = new int[N];

        // Fill the arrays 'a' and 'b' on the CPU
        for (int i = 0; i < N; i++) {
            a[i] = -i;
            b[i] = i * i;
        }

        // Allocate memory and copy data to the GPU
        var deviceA = new CUdeviceptr();
        cuMemAlloc(deviceA, SIZE);
        cuMemcpyHtoD(deviceA, Pointer.to(a), SIZE);

        var deviceB = new CUdeviceptr();
        cuMemAlloc(deviceB, SIZE);
        cuMemcpyHtoD(deviceB, Pointer.to(b), SIZE);

        var deviceC = new CUdeviceptr();
        cuMemAlloc(deviceC, SIZE);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        var kernelParameters = Pointer.to(
                Pointer.to(deviceA),
                Pointer.to(deviceB),
                Pointer.to(deviceC),
                Pointer.to(new int[]{N})
        );

        // Call the kernel function
        var blockSizeX = 1;
        cuLaunchKernel(function,
                N, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,     // Block dimension
                0, null,              // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Copy the result back to host
        cuMemcpyDtoH(Pointer.to(c), deviceC, SIZE);

        // Display the results
        IntStream.range(0, N).forEach(i -> printf("%d + %d = %d", a[i], b[i], c[i]));

        // Clean up
        cuMemFree(deviceA);
        cuMemFree(deviceB);
        cuMemFree(deviceC);
        cuCtxDestroy(context);
    }
}
