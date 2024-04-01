package org.jcuda.kacygan.chapter3;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

public class JCudaCode {
    private static final Logger logger = LogManager.getLogger();
    private static final Random random = new Random();

    public static void printf(String formatString, Object... params) {
        logger.printf(Level.DEBUG, formatString, params);
    }

    public static List<CUdeviceptr> allocateAndCopyMemory(Pointer hostA, Pointer hostB, long size) {
        var deviceList = new ArrayList<CUdeviceptr>();

        var devA = new CUdeviceptr();
        var devB = new CUdeviceptr();
        var devC = new CUdeviceptr();
        cuMemAlloc(devA, size);
        cuMemAlloc(devB, size);
        cuMemAlloc(devC, size);
        cuMemcpyHtoD(devA, hostA, size);
        cuMemcpyHtoD(devB, hostB, size);

        deviceList.add(devA);
        deviceList.add(devB);
        deviceList.add(devC);

        return deviceList;
    }

    public static byte[] generateRandomBlock(int size) {
        var data = new byte[size];
        random.nextBytes(data);
        return data;
    }
}
