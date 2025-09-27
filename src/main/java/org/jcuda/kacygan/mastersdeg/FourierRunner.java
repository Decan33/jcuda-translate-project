package org.jcuda.kacygan.mastersdeg;

import java.util.List;

public class FourierRunner {

    public static void main(String[] args) {
        System.out.println("Running Fourier tests with cold runs");
        var tests = List.of(
//                new FourierCalculatorRaw(),
//                new FourierCalculatorOptimized(),
//                new FourierCalculatorPinned(),
//                new FourierCalculatorStreams(),
//                new FourierCalculatorStreamsAndShared()
                new JCudaMemoryTest(MemoryType.NORMAL),
                new JCudaMemoryTest(MemoryType.PINNED)
        );

        for (var test : tests) {
            test.runTest();
        }
    }
}
