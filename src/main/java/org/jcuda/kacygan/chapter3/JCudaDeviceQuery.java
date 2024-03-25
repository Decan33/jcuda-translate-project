package org.jcuda.kacygan.chapter3;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaGetDeviceProperties;

public class JCudaDeviceQuery {
    public static final Logger logger = LogManager.getLogger();

    public static void main(String[] args) {
        // Initialize JCuda
        JCuda.cudaSetDevice(0);

        // Get the number of devices
        int[] count = new int[1];
        cudaGetDeviceCount(count);

        for (int i = 0; i < count[0]; i++) {
            cudaDeviceProp prop = new cudaDeviceProp();
            cudaGetDeviceProperties(prop, i);

            logger.printf(Level.DEBUG, "   --- General Information for device %d ---", i);
            logger.printf(Level.DEBUG, "Name: %s",  new String(prop.name));
            logger.printf(Level.DEBUG, "Compute capability: %d.%d", prop.major, prop.minor);
            logger.printf(Level.DEBUG, "Clock rate: %d", prop.clockRate);
            logger.printf(Level.DEBUG, "Device copy overlap: %s", (prop.deviceOverlap == 1 ? "Enabled" : "Disabled"));
            logger.printf(Level.DEBUG, "Kernel execution timeout : %s", (prop.kernelExecTimeoutEnabled == 1 ? "Enabled" : "Disabled"));

            logger.printf(Level.DEBUG, "   --- Memory Information for device %d ---", i);
            logger.printf(Level.DEBUG, "Total global mem: %d",  prop.totalGlobalMem);
            logger.printf(Level.DEBUG, "Total constant Mem: %d", prop.totalConstMem);
            logger.printf(Level.DEBUG, "Max mem pitch: %d", prop.memPitch);
            logger.printf(Level.DEBUG, "Texture Alignment: %d", prop.textureAlignment);

            logger.printf(Level.DEBUG, "   --- MP Information for device %d ---", i);
            logger.printf(Level.DEBUG, "Multiprocessor count: %d", prop.multiProcessorCount);
            logger.printf(Level.DEBUG, "Shared mem per mp: %d", prop.sharedMemPerBlock);
            logger.printf(Level.DEBUG, "Registers per mp: %d", prop.regsPerBlock);
            logger.printf(Level.DEBUG, "Threads in warp: %d", prop.warpSize);
            logger.printf(Level.DEBUG, "Max threads per block: %d", prop.maxThreadsPerBlock);
            logger.printf(Level.DEBUG, "Max thread dimensions: (%d, %d, %d)", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            logger.printf(Level.DEBUG, "Max grid dimensions: (%d, %d, %d)", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        }
    }
}
