package org.jcuda.kacygan.chapter3;

import jcuda.runtime.cudaDeviceProp;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import static jcuda.runtime.JCuda.cudaChooseDevice;
import static jcuda.runtime.JCuda.cudaGetDevice;
import static jcuda.runtime.JCuda.cudaSetDevice;

public class ChooseAndSetDevice {
    // set_gpu.cu
    private static final Logger logger = LogManager.getLogger();

    public static void main(String[] args) {
        var dev = new int[1];
        cudaGetDevice(dev);
        logger.printf(Level.DEBUG, "ID of current CUDA device: %d%n", dev[0]);

        var prop = new cudaDeviceProp();
        prop.major = 1;
        prop.minor = 3;
        cudaChooseDevice(dev, prop);
        logger.printf(Level.DEBUG, "ID of CUDA device closest to revision 1.3: %d%n", dev[0]);

        cudaSetDevice(dev[0]);
    }
}
