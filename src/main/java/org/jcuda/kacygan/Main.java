package org.jcuda.kacygan;


import jcuda.runtime.JCuda;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jcuda.kacygan.util.JCudaMemoryHandler;

public class Main {
    public static final Logger logger = LogManager.getLogger();

    public static void main(String[] args) {
        JCuda.setExceptionsEnabled(true);

        var chunkOne = JCudaMemoryHandler.generateRandomByteArrayWith(69);
        var chunkTwo = JCudaMemoryHandler.generateRandomArrayWith(420);

        chunkOne[1] = 12;
        chunkTwo[0] = 47;

        logger.info("Hello world!");
    }
}