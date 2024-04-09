package org.jcuda.kacygan.chapter10;

import org.jcuda.kacygan.constants.CudaConstants;

import java.util.Random;

public class BasicDoubleStream {
    public static void main(String[] args) {
        var hostA = new int[CudaConstants.CHAPTER_10_FULL_DATA_SIZE.size];
        var hostB = new int[CudaConstants.CHAPTER_10_FULL_DATA_SIZE.size];
        var hostC = new int[CudaConstants.CHAPTER_10_FULL_DATA_SIZE.size];

        var random = new Random();
        for (int i = 0; i < CudaConstants.CHAPTER_10_FULL_DATA_SIZE.size; i++) {
            hostA[i] = random.nextInt();
            hostB[i] = random.nextInt();
        }

        var cudaOps = new CudaDoubleStreamBasic("DoubleStream.ptx");
        cudaOps.performComputation(hostA, hostB, hostC);

        // Timing and verification logic omitted for brevity
    }
}

