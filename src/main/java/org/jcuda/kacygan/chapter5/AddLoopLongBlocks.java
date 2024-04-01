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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

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
import static jcuda.driver.JCudaDriver.cuModuleLoadData;

public class AddLoopLongBlocks extends JCudaCode {
    private static final int N = 33 * 1024;
    private static final Long SIZE = (long) (N * Sizeof.INT);
    private static final int BLOCK_SIZE = 128;
    private static final String PTX_FILE = "AddKernel.ptx";

    public static void main(String[] args) throws IOException {
        // Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        // Prepare PTX and function from the compiled CUDA kernel
        var ptxSource = new String(Files.readAllBytes(Paths.get(PTX_FILE)));
        var module = new CUmodule();
        cuModuleLoadData(module, ptxSource);
        var function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        // Allocate memory on the host (CPU)
        var hostA = new int[N];
        var hostB = new int[N];
        var hostC = new int[N];
        for (var i = 0; i < N; i++) {
            hostA[i] = i;
            hostB[i] = 2 * i;
        }

        // Allocate and copy memory to the device (GPU)
        var devices = allocateAndCopyMemory(Pointer.to(hostA), Pointer.to(hostB), SIZE);
        var devA = devices.get(0);
        var devB = devices.get(1);
        var devC = devices.get(2);

        // Set up the kernel parameters and launch the kernel
        Pointer kernelParameters = Pointer.to(Pointer.to(devA), Pointer.to(devB), Pointer.to(devC));
        cuLaunchKernel(function,
                N / BLOCK_SIZE, 1, 1,  // Grid dimension
                BLOCK_SIZE, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        // Copy the result back to the host
        cuMemcpyDtoH(Pointer.to(hostC), devC, SIZE);

        // Verify the results
        var success = true;
        for (var i = 0; i < N; i++) {
            if (hostA[i] + hostB[i] != hostC[i]) {
                printf("Error: %d + %d != %d\n", hostA[i], hostB[i], hostC[i]);
                success = false;
                break;
            }
        }
        if (success) System.out.println("We did it!");

        // Clean up
        cuMemFree(devA);
        cuMemFree(devB);
        cuMemFree(devC);
        cuCtxDestroy(context);
    }
}
