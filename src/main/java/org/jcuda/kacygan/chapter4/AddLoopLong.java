package org.jcuda.kacygan.chapter4;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

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

public class AddLoopLong {
    private static String KERNEL_SOURCE_CODE = """
            extern "C"
            __global__ void add( int *a, int *b, int *c ) {
                int tid = blockIdx.x;    // this thread handles the data at its thread id
                if (tid < N)
                    c[tid] = a[tid] + b[tid];
            }
            """;

    private static boolean COMPILATION_AS_STRING = true;
    private static final Integer N = 32 * 1024;
    private static final long SIZE = N * Sizeof.INT;

    public static void main(String[] args) {
        // Initialize the driver and create a context
        cuInit(0);
        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Load the kernel from the PTX file
        var ptxFileName = "AddKernel.ptx";
        var module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        var function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        var hostA = new int[N];
        var hostB = new int[N];
        var hostC = new int[N];

        // Initialize the host arrays
        for (var i = 0; i < N; i++) {
            hostA[i] = i;
            hostB[i] = 2 * i;
        }

        // Allocate and copy arrays to the device
        var deviceA = new CUdeviceptr();
        var deviceB = new CUdeviceptr();
        var deviceC = new CUdeviceptr();
        cuMemAlloc(deviceA, SIZE);
        cuMemAlloc(deviceB, SIZE);
        cuMemAlloc(deviceC, SIZE);
        cuMemcpyHtoD(deviceA, Pointer.to(hostA), SIZE);
        cuMemcpyHtoD(deviceB, Pointer.to(hostB), SIZE);

        // Kernel parameters
        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceA),
                Pointer.to(deviceB),
                Pointer.to(deviceC),
                Pointer.to(new int[]{N})
        );

        // Launch the kernel
        cuLaunchKernel(function,
                128, 1, 1,  // Grid dimension
                1, 1, 1,   // Block dimension
                0, null,   // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Copy the result back to the host
        cuMemcpyDtoH(Pointer.to(hostC), deviceC, SIZE);

        // Verify and print result
        var success = true;
        for (var i = 0; i < N; i++) {
            if (hostA[i] + hostB[i] != hostC[i]) {
                System.out.printf("Error: %d + %d != %d\n", hostA[i], hostB[i], hostC[i]);
                success = false;
                break;
            }
        }
        if (success) System.out.println("We did it!");

        // Clean up
        cuMemFree(deviceA);
        cuMemFree(deviceB);
        cuMemFree(deviceC);
        cuCtxDestroy(context);
    }
}
