package org.jcuda.kacygan.chapter9;

public class HistogramCpuCalculator {

    public static int[] calculateHistogram(byte[] data) {
        var histogram = new int[256];
        for (var value : data) {
            histogram[value & 0xFF]++;
        }
        return histogram;
    }
}
