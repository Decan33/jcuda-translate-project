package org.jcuda.kacygan.chapter9;

public class Timer {
    private long startTime;
    private long endTime;

    public void start() {
        startTime = System.nanoTime();
    }

    public void stop() {
        endTime = System.nanoTime();
    }

    public double elapsedTimeMillis() {
        return (endTime - startTime) / 1_000_000.0;
    }
}

