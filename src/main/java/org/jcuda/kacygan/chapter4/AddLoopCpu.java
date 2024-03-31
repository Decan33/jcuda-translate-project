package org.jcuda.kacygan.chapter4;

import org.jcuda.kacygan.chapter3.JCudaCode;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.IntStream;

public class AddLoopCpu extends JCudaCode {
    private static final Integer N = 10;
    public static void main(String[] args) {
        var a = new Integer[N];
        var b = new Integer[N];

        IntStream.range(0, N).forEach(
                num -> {
                    a[num] = -num;
                    b[num] = num * num;
                }
        );

        var c = add(a, b);

        IntStream.range(0, N).forEach(
                num -> printf("%d + %d = %d%n", a[num], b[num], c[num])
        );
    }

    private static Integer[] add( Integer[] a, Integer[] b) {
        var c = new Integer[N];
        var tid = 0;

        while(tid < N) {
            c[tid] = a[tid] + b[tid];
            tid += 1;
        }

        return c;
    }
}
