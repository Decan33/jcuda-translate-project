package org.jcuda.kacygan.chapter9;

import org.jcuda.kacygan.chapter3.JCudaCode;

public class HistogramGpuSharedMemory extends JCudaCode {
    private static final int SIZE = 100 * 1024 * 1024;
    private static final Timer timer = new Timer();

    public static void main(String[] args) {
        var data = generateRandomBlock(SIZE);

        timer.start();
        HistogramGpuSharedMemoryCalculator.calculateHistogram(data);
        timer.stop();
        var elapsedTime = timer.elapsedTimeMillis();

        System.out.printf("Time to generate: %3.1f seconds\n", elapsedTime);
    }
}
