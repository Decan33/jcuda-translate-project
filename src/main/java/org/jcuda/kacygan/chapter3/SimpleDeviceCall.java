package org.jcuda.kacygan.chapter3;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.nvrtc.nvrtcProgram;
import jcuda.runtime.JCuda;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;


public class SimpleDeviceCall extends JCudaCode {

    /**
     *
     * An example showing how to use the NVRTC (NVIDIA Runtime Compiler) API
     * to compile CUDA kernel code at runtime.
     * The source code of the program that will be compiled at runtime:
     * A simple vector addition kernel.
     * Note: The function should be declared as
     * extern "C"
     * to make sure that it can be found under the given name.
     */
    private static String KERNEL_SOURCE_CODE = """
            extern "C"
            __global__ void add( int a, int b, int* c) {
                *c = addem( a, b );
            }
            """;

    private static String KERNEL_FUNCTION_CALL = """
            extern "C"
            __device int addem( int a, int b ) {
                return a + b;
            }
            """;
    private static boolean COMPILATION_AS_STRING = true;

    public static void main(String[] args) {
        // Initialize the driver and create a context

        cuInit(0);
        var context = new CUcontext();
        var device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);

        if (COMPILATION_AS_STRING) {
            var program = new nvrtcProgram();
            nvrtcCreateProgram(
                    program, KERNEL_SOURCE_CODE, null, 0, null, null
            );

            nvrtcCompileProgram(program, 0, null);

            var logs = new String[1];
            nvrtcGetProgramLog(program, logs);
            printf("Logs of compilated program: %d", logs[0]);

            var ptxCode = new String[1];
            nvrtcGetPTX(program, ptxCode);
            nvrtcDestroyProgram(program);

            var cuModule = new CUmodule();
            cuModuleLoadData(cuModule, ptxCode[0]);

            var cuFunction = new CUfunction();
            cuModuleGetFunction(cuFunction, cuModule, "add");

            var a = 2;
            var b = 7;
            var hostC = new int[1];
            var deviceC = new CUdeviceptr();
            cuMemAlloc(deviceC, Sizeof.INT);

            var kernelParams = Pointer.to(
                    Pointer.to(new int[]{a}),
                    Pointer.to(new int[]{b}),
                    Pointer.to(deviceC)
            );

            // Launch the kernel
            cuLaunchKernel(cuFunction,
                    1, 1, 1,      // Grid dimension
                    1, 1, 1,      // Block dimension
                    0, null,      // Shared memory size and stream
                    kernelParams, null // Kernel- and extra parameters
            );

            // Copy the result from the device to the host
            cuMemcpyDtoH(Pointer.to(hostC), deviceC, Sizeof.INT);
            printf("2 + 7 = %d", hostC[0]);

            // Clean up
            cuMemFree(deviceC);
            cuCtxDestroy(context);
            return;
        }

        // Load the kernel from the PTX file
        String ptxFileName = "AddKernel.ptx";
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "add");

        int a = 2;
        int b = 7;
        int[] hostC = new int[1];
        CUdeviceptr deviceC = new CUdeviceptr();
        cuMemAlloc(deviceC, Sizeof.INT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{a}),
                Pointer.to(new int[]{b}),
                Pointer.to(deviceC)
        );

        // Launch the kernel
        cuLaunchKernel(function,
                1, 1, 1,      // Grid dimension
                1, 1, 1,      // Block dimension
                0, null,      // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize(); // Wait for the kernel to complete

        // Copy the result from the device to the host
        cuMemcpyDtoH(Pointer.to(hostC), deviceC, Sizeof.INT);
        printf("2 + 7 = %d", hostC[0]);

        // Clean up
        cuMemFree(deviceC);
        cuCtxDestroy(context);
    }
}
