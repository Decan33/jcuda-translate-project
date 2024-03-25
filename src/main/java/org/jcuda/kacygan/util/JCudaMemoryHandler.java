package org.jcuda.kacygan.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Random;
import java.util.stream.IntStream;

public class JCudaMemoryHandler {
    public static final Logger logger = LogManager.getLogger();
    private static final Random random = new Random();

    private JCudaMemoryHandler() {

    }

    public static byte[] generateRandomByteArrayWith(int size) {
        logger.info("Generating random block of size {}", size);
        byte[] data = new byte[size];
        random.nextBytes(data);
        return data;
    }

    public static int[] generateRandomArrayWith(int size) {
        logger.info("Generating random int array block of size {}", size);
        return IntStream.range(0, size).map(i -> random.nextInt()).toArray();
    }





}
