package org.jcuda.kacygan.mastersdeg;/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2013 Marco Hutter - http://www.jcuda.org
 */
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaEventCreate;
import static jcuda.runtime.JCuda.cudaEventDestroy;
import static jcuda.runtime.JCuda.cudaEventElapsedTime;
import static jcuda.runtime.JCuda.cudaEventRecord;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaFreeHost;
import static jcuda.runtime.JCuda.cudaHostAlloc;
import static jcuda.runtime.JCuda.cudaHostAllocWriteCombined;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpyAsync;
import static jcuda.runtime.JCuda.cudaSetDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.nio.ByteBuffer;
import java.util.Locale;

import jcuda.Pointer;
import jcuda.runtime.cudaEvent_t;

/**
 * A test for the bandwidth of various copying operations. 
 *
 * This test computes the bandwidth of the data transfer from the host to 
 * the device. The host data is once allocated as pinned memory 
 * (using cudaHostAlloc) and once as pageable memory (a Java array or 
 * a direct buffer, comparable to 'malloc' in C).
 */
public class JCudaBandwidthTest
{
    static final int NUM_REPS = 100; // Should match data.h

    /**
     * Memory modes for the host memory
     */
    enum HostMemoryMode
    {
        /**
         * Pinned host memory, allocated with cudaHostAlloc
         */
        PINNED,

        /**
         * Pageable memory in form of a Pointer.to(array)
         */
        PAGEABLE_ARRAY,

        /**
         * Pageable memory in form of a Pointer.to(directBuffer)
         */
        PAGEABLE_DIRECT_BUFFER,
    }

    /**
     * Entry point of this test
     *
     * @param args Not used
     */
    public static void main(String[] args)
    {
        int device = 0;
        cudaSetDevice(device);

        int hostAllocFlags = cudaHostAllocWriteCombined;
        runTest(HostMemoryMode.PINNED, hostAllocFlags);
        runTest(HostMemoryMode.PAGEABLE_ARRAY, hostAllocFlags);
        runTest(HostMemoryMode.PAGEABLE_DIRECT_BUFFER, hostAllocFlags);

        System.out.println("Done");
    }


    /**
     * Run a test that computes the bandwidth for copying host memory to the 
     * device, using various memory block sizes, and print the results
     *
     * @param hostMemoryMode The {@link HostMemoryMode}
     * @param hostAllocFlags The flags for cudaHostAlloc
     */
    static void runTest(HostMemoryMode hostMemoryMode, int hostAllocFlags)
    {
        int minExponent = 10;
        int maxExponent = 28;
        int count = maxExponent - minExponent;
        int[] memorySizes = new int[count];
        float[] timesMs = new float[memorySizes.length];
        float[] mbps = new float[memorySizes.length];

        System.out.print("Running");
        for (int i=0; i<count; i++)
        {
            System.out.print(".");
            memorySizes[i] = (1 << minExponent + i);
            float[] result = computeCopyTimeAndBandwidth(
                    hostMemoryMode, hostAllocFlags, memorySizes[i]);
            timesMs[i] = result[0];
            mbps[i] = result[1];
        }
        System.out.println();

        System.out.printf("%12s | %15s | %15s\n", "Size (bytes)", "Time (s)", "Bandwidth (MB/s)");
        System.out.println("---------------------------------------------------------------");
        for (int i=0; i<memorySizes.length; i++)
        {
            String s = String.format("%12d", memorySizes[i]);
            String t = String.format(Locale.ENGLISH, "%15.6f", timesMs[i] / 1000.0f);
            String b = String.format(Locale.ENGLISH, "%15.3f", mbps[i]);
            System.out.println(s+" | "+t+" | "+b);
        }
        System.out.println("\n");
    }


    /**
     * Compute the time in milliseconds and bandwidth in MBps for copying data from the
     * host to the device
     *
     * @param hostMemoryMode The {@link HostMemoryMode}
     * @param hostAllocFlags The flags for the cudaHostAlloc call
     * @param memorySize The memory size, in bytes
     * @return float[]{elapsedTimeMs, bandwidthMBps}
     */
    static float[] computeCopyTimeAndBandwidth(
            HostMemoryMode hostMemoryMode, int hostAllocFlags, int memorySize)
    {
        // Initialize the host memory
        Pointer hostData;
        ByteBuffer hostDataBuffer;
        if (hostMemoryMode == HostMemoryMode.PINNED)
        {
            // Allocate pinned (page-locked) host memory
            hostData = new Pointer();
            cudaHostAlloc(hostData, memorySize, hostAllocFlags);
            hostDataBuffer = hostData.getByteBuffer(0, memorySize);
        }
        else if (hostMemoryMode == HostMemoryMode.PAGEABLE_ARRAY)
        {
            // The host memory is pageable and stored in a Java array
            var array = new byte[memorySize];
            hostDataBuffer = ByteBuffer.wrap(array);
            hostData = Pointer.to(array);
        }
        else
        {
            // The host memory is pageable and stored in a direct byte buffer
            hostDataBuffer = ByteBuffer.allocateDirect(memorySize);
            hostData = Pointer.to(hostDataBuffer);
        }

        // Fill the memory with arbitrary data
        for (int i = 0; i < memorySize; i++)
        {
            hostDataBuffer.put(i, (byte)i);
        }

        // Allocate device memory
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, memorySize);

        float elapsedTimeMs = computeCopyTime(
                deviceData, hostData, cudaMemcpyHostToDevice, memorySize, NUM_REPS);
        float totalBytes = (float)memorySize * NUM_REPS;
        float bandwidthMBps = (totalBytes / (elapsedTimeMs / 1000.0f)) / (1024.0f * 1024.0f);

        // Clean up
        if (hostMemoryMode == HostMemoryMode.PINNED)
        {
            cudaFreeHost(hostData);
        }
        cudaFree(deviceData);
        return new float[]{elapsedTimeMs, bandwidthMBps};
    }

    /**
     * Compute the time in milliseconds for copying data from the 
     * given source pointer to the given destination pointer
     *
     * @param dstData The destination pointer
     * @param srcData The source pointer
     * @param memcopyKind The cudaMemcpyKind. Must match the types 
     * of the source and destination pointers!
     * @param memSize The memory size, in bytes
     * @param runs The number of times that the copying operation
     * should be repeated
     * @return The time in milliseconds
     */
    static float computeCopyTime(
            Pointer dstData, Pointer srcData,
            int memcopyKind, int memSize, int runs)
    {
        // Initialize the events for the time measure
        cudaEvent_t start = new cudaEvent_t();
        cudaEvent_t stop = new cudaEvent_t();
        cudaEventCreate(start);
        cudaEventCreate(stop);

        // Perform the specified number of copying operations
        cudaEventRecord(start, null);
        for (int i = 0; i < runs; i++)
        {
            cudaMemcpyAsync(dstData, srcData, memSize, memcopyKind, null);
        }
        cudaEventRecord(stop, null);
        cudaDeviceSynchronize();

        // Compute the elapsed time
        float[] elapsedTimeMsArray = { Float.NaN };
        cudaEventElapsedTime(elapsedTimeMsArray, start, stop);
        float elapsedTimeMs = elapsedTimeMsArray[0];

        // Clean up
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return elapsedTimeMs;
    }
}