package org.jcuda.kacygan.mastersdeg;

import java.util.List;

public class FourierRunner {

    public static void main(String[] args) {
        System.out.println("Running Fourier tests with cold runs");
        var tests = List.of(
                new FourierCalculator(),
                new FourierCalculator2(),
                new FourierCalculator3(),
                new FourierCalculator4()
        );

        for (var test : tests) {
            test.runTest();
        }
    }
}
