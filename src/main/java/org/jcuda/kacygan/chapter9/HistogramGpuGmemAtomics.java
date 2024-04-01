package org.jcuda.kacygan.chapter9;

import org.jcuda.kacygan.chapter3.JCudaCode;

import java.util.Arrays;

public class HistogramGpuGmemAtomics extends JCudaCode {
    private static final int SIZE = 100 * 1024 * 1024;
    private static final Timer timer = new Timer();

    public static void main(String[] args) {
        var data = generateRandomBlock(SIZE);

        timer.start();
        var histogram = HistogramGpuCalculator.calculateHistogram(data);
        timer.stop();

        var elapsedTime = timer.elapsedTimeMillis();
        System.out.printf("Time to generate: %3.1f seconds\n", elapsedTime);

        var histogramCount = Arrays
                .stream(histogram)
                .asLongStream()
                .sum();

        System.out.println("Histogram Sum: " + histogramCount);
    }
}
