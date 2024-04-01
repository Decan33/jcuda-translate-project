package org.jcuda.kacygan.chapter9;

import org.jcuda.kacygan.chapter3.JCudaCode;

import java.util.Arrays;
import java.util.Random;

public class HistogramCpu extends JCudaCode {
    private static final int SIZE = 100 * 1024 * 1024;

    public static void main(String[] args) {
        var buffer = generateRandomBlock(SIZE);

        var timer = new Timer();
        timer.start();

        var histogram = HistogramCpuCalculator.calculateHistogram(buffer);

        timer.stop();
        System.out.printf("Time to generate: %3.1f ms\n", timer.elapsedTimeMillis());

        var histogramCount = Arrays
                .stream(histogram)
                .asLongStream()
                .sum();
        System.out.println("Histogram Sum: " + histogramCount);
    }
}

