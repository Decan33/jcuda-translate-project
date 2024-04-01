package org.jcuda.kacygan.chapter5;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
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

public class AddLoopBlocks extends JCudaCode {
    private static final Integer N = 10;
    private static final long SIZE = N * Sizeof.INT;
    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Load the PTX file
        var module = new CUmodule();
        cuModuleLoad(module, "VectorAddKernel.ptx");
        var function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        var hostA = new int[N];
        var hostB = new int[N];
        var hostC = new int[N];

        // Initialize the arrays 'a' and 'b'
        for (var i = 0; i < N; i++) {
            hostA[i] = i;
            hostB[i] = i * i;
        }

        // Allocate and copy memory to the GPU
        var devices = allocateAndCopyMemory(Pointer.to(hostA), Pointer.to(hostB), SIZE);
        var devA = devices.get(0);
        var devB = devices.get(1);
        var devC = devices.get(2);

        // Execute the kernel
        var kernelParameters = Pointer.to(
                Pointer.to(devA),
                Pointer.to(devB),
                Pointer.to(devC)
        );
        cuLaunchKernel(function,
                1, 1, 1, // Grid dimension
                N, 1, 1, // Block dimension
                0, null, // Shared memory size
                kernelParameters, null // Kernel parameters
        );
        cuCtxSynchronize();

        // Copy the result back to host memory
        cuMemcpyDtoH(Pointer.to(hostC), devC, SIZE);

        IntStream.range(0, N).forEach(i -> printf("%d + %d = %d\n", hostA[i], hostB[i], hostC[i]));

        // Clean up
        cuMemFree(devA);
        cuMemFree(devB);
        cuMemFree(devC);
        cuCtxDestroy(context);
    }
}
