package org.jcuda.kacygan.chapter10;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.cudaMalloc;

import jcuda.*;
import jcuda.driver.*;
import org.jcuda.kacygan.constants.CudaConstants;

public class CopyTimed {

    static {
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        var device = new CUdevice();
        var context = new CUcontext();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);
    }

    public static float cudaMallocTest(int size, boolean up) {
        var startEvent = new CUevent();
        var stopEvent = new CUevent();
        cuEventCreate(startEvent, CUevent_flags.CU_EVENT_DEFAULT);
        cuEventCreate(stopEvent, CUevent_flags.CU_EVENT_DEFAULT);

        var hostData = new int[size];
        var deviceData = new CUdeviceptr();
        cudaMalloc(deviceData, (long) size * Sizeof.INT);

        cuEventRecord(startEvent, null);
        for (int i = 0; i < 100; i++) {
            if (up) {
                cuMemcpyHtoD(deviceData, Pointer.to(hostData), (long) size * Sizeof.INT);
            } else {
                cuMemcpyDtoH(Pointer.to(hostData), deviceData, (long) size * Sizeof.INT);
            }
        }
        cuEventRecord(stopEvent, null);
        cuEventSynchronize(stopEvent);

        float[] elapsedTime = new float[1];
        cuEventElapsedTime(elapsedTime, startEvent, stopEvent);

        cuMemFree(deviceData);
        cuEventDestroy(startEvent);
        cuEventDestroy(stopEvent);

        return elapsedTime[0];
    }

    public static float cudaHostAllocTest(int size, boolean up) {
        var startEvent = new CUevent();
        var stopEvent = new CUevent();
        cuEventCreate(startEvent, CUevent_flags.CU_EVENT_DEFAULT);
        cuEventCreate(stopEvent, CUevent_flags.CU_EVENT_DEFAULT);

        var hostData = new Pointer();
        var deviceData = new CUdeviceptr();
        cuMemHostAlloc(hostData, (long) size * Sizeof.INT, CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL);
        cudaMalloc(deviceData, (long) size * Sizeof.INT);

        cuEventRecord(startEvent, null);
        for (int i = 0; i < 100; i++) {
            if (up) {
                cuMemcpyHtoD(deviceData, hostData, (long) size * Sizeof.INT);
            } else {
                cuMemcpyDtoH(hostData, deviceData, (long) size * Sizeof.INT);
            }
        }
        cuEventRecord(stopEvent, null);
        cuEventSynchronize(stopEvent);

        float[] elapsedTime = new float[1];
        cuEventElapsedTime(elapsedTime, startEvent, stopEvent);

        cuMemFreeHost(hostData);
        cuMemFree(deviceData);
        cuEventDestroy(startEvent);
        cuEventDestroy(stopEvent);

        return elapsedTime[0];
    }

    public static void main(String[] args) {
        float elapsedTime;
        float megabytes = (float)100 * CudaConstants.CHAPTER_10_COPY_TIMED_TEST.size * Sizeof.INT / 1024 / 1024;

        elapsedTime = cudaMallocTest(CudaConstants.CHAPTER_10_COPY_TIMED_TEST.size, true);
        System.out.printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
        System.out.printf("\tMB/s during copy up: %3.1f\n", megabytes / (elapsedTime / 1000));

        elapsedTime = cudaMallocTest(CudaConstants.CHAPTER_10_COPY_TIMED_TEST.size, false);
        System.out.printf("Time using cudaMalloc: %3.1f ms\n", elapsedTime);
        System.out.printf("\tMB/s during copy down: %3.1f\n", megabytes / (elapsedTime / 1000));

        elapsedTime = cudaHostAllocTest(CudaConstants.CHAPTER_10_COPY_TIMED_TEST.size, true);
        System.out.printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
        System.out.printf("\tMB/s during copy up: %3.1f\n", megabytes / (elapsedTime / 1000));

        elapsedTime = cudaHostAllocTest(CudaConstants.CHAPTER_10_COPY_TIMED_TEST.size, false);
        System.out.printf("Time using cudaHostAlloc: %3.1f ms\n", elapsedTime);
        System.out.printf("\tMB/s during copy down: %3.1f\n", megabytes / (elapsedTime / 1000));
    }
}

