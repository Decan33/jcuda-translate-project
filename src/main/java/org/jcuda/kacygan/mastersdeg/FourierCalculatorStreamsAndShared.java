
package org.jcuda.kacygan.mastersdeg;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUevent;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

import static jcuda.driver.CUstream_flags.CU_STREAM_NON_BLOCKING;
import static jcuda.driver.JCudaDriver.*;

/**
 * Streams + constant memory JCuda implementation.
 * Kernel stays untouched and is expected to be named "fourier" (see FourierTest.FUNCTION_NAME).
 */
@SuppressWarnings("java:S106")
public class FourierCalculatorStreamsAndShared implements FourierTest {

    private static final String PTX_FILENAME = "FourierOptimized.ptx";

    // ===== Entry point =====
    @Override
    public void runTest() {
        System.out.println("Running Fourier (streams + constants + shared). Kernel: " + FUNCTION_NAME);
        JCudaDriver.setExceptionsEnabled(true);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, PTX_FILENAME);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, FUNCTION_NAME);

        ConstantSymbols consts = ConstantSymbols.resolve(module);
        consts.uploadStatic(); // Upload everything except tmin

        // Cold run (warm-up)
        System.out.println("Cold run...");
        runStreamsOnce(function, consts, /*writeResults*/ false);
        System.out.println("Cold run done.");

        // Timings
        Stats sPrep = new Stats(NUM_REPS);
        Stats sKernel = new Stats(NUM_REPS);
        Stats sCopy = new Stats(NUM_REPS);
        Stats sDealloc = new Stats(NUM_REPS);
        Stats sTotal = new Stats(NUM_REPS);

        for (int rep = 0; rep < NUM_REPS; rep++) {
            if (logReps) System.out.println("Rep " + (rep + 1) + " / " + NUM_REPS);
            double[] ms = runStreamsOnce(function, consts, /*writeResults*/ rep == NUM_REPS - 1);
            sPrep.add(ms[0]);
            sKernel.add(ms[1]);
            sCopy.add(ms[2]);
            sDealloc.add(ms[3]);
            sTotal.add(ms[4]);
        }

        System.out.printf("Prep:    %.3f ms \u00B1 %.3f%n", sPrep.mean() / 1000.0, sPrep.std());
        System.out.printf("Kernel:  %.3f ms \u00B1 %.3f (wall)%n", sKernel.mean()/ 1000.0, sKernel.std());
        System.out.printf("Copy:    %.3f ms \u00B1 %.3f (wall)%n", sCopy.mean()/ 1000.0, sCopy.std());
        System.out.printf("Dealloc: %.3f ms \u00B1 %.3f%n", sDealloc.mean()/ 1000.0, sDealloc.std());
        System.out.printf("Total:   %.3f ms \u00B1 %.3f%n", sTotal.mean()/ 1000.0, sTotal.std());

        cuModuleUnload(module);
        cuCtxDestroy(context);
        System.out.println("Finished.");
    }

    /**
     * One full pass using NUM_STREAMS async streams.
     * @return double[5] = {prepMs, kernelWallMs, copyWallMs, deallocMs, totalMs}
     */
    private double[] runStreamsOnce(CUfunction function, ConstantSymbols consts, boolean writeResults) {
        long t0 = System.nanoTime();

        // Allocate & prepare per stream
        CUstream[] streams = new CUstream[NUM_STREAMS];
        CUdeviceptr[] dOut = new CUdeviceptr[NUM_STREAMS];
        Pointer[] hPinned = new Pointer[NUM_STREAMS];
        FloatBuffer[] hBuf = new FloatBuffer[NUM_STREAMS];
        int[] currentSizes = new int[NUM_STREAMS];
        int[] allocCounts = new int[NUM_STREAMS];

        for (int i = 0; i < NUM_STREAMS; i++) {
            streams[i] = new CUstream();
            cuStreamCreate(streams[i], CU_STREAM_NON_BLOCKING);

            int startIdx = i * CHUNK_SIZE;
            int endIdx = Math.min(startIdx + CHUNK_SIZE, LENGTH);
            int cur = Math.max(0, endIdx - startIdx);
            currentSizes[i] = cur;

            int blocks = gridFor(cur);
            int alloc = Math.max(1, blocks * THREADS_PER_BLOCK); // avoid 0-length alloc
            allocCounts[i] = alloc;

            dOut[i] = new CUdeviceptr();
            cuMemAlloc(dOut[i], (long) alloc * Sizeof.FLOAT);

            hPinned[i] = new Pointer();
            cuMemHostAlloc(hPinned[i], (long) Math.max(1, cur) * Sizeof.FLOAT, 0);

            hBuf[i] = hPinned[i]
                    .getByteBuffer(0, (long) Math.max(1, cur) * Sizeof.FLOAT)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }

        long tPrepEnd = System.nanoTime();

        // Events for timing
        CUevent[] evKernelStart = new CUevent[NUM_STREAMS];
        CUevent[] evKernelEnd = new CUevent[NUM_STREAMS];
        CUevent[] evCopyEnd = new CUevent[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            evKernelStart[i] = new CUevent(); cuEventCreate(evKernelStart[i], 0);
            evKernelEnd[i]   = new CUevent(); cuEventCreate(evKernelEnd[i], 0);
            evCopyEnd[i]     = new CUevent(); cuEventCreate(evCopyEnd[i], 0);
        }

        // Launch + async copies
        for (int i = 0; i < NUM_STREAMS; i++) {
            int startIdx = i * CHUNK_SIZE;
            int cur = currentSizes[i];
            if (cur == 0) continue;

            float chunkTmin = TMIN + startIdx * DELTA;
            consts.setTmin(chunkTmin);

            int blocks = gridFor(cur);
            int sharedMemBytes = COEFFICIENTS * Sizeof.FLOAT;
            Pointer kernelParams = Pointer.to(Pointer.to(dOut[i]));

            cuEventRecord(evKernelStart[i], streams[i]);
            cuLaunchKernel(function,
                    blocks, 1, 1,
                    THREADS_PER_BLOCK, 1, 1,
                    sharedMemBytes,
                    streams[i],
                    kernelParams, null);
            cuEventRecord(evKernelEnd[i], streams[i]);

            cuMemcpyDtoHAsync(hPinned[i], dOut[i], (long) cur * Sizeof.FLOAT, streams[i]);
            cuEventRecord(evCopyEnd[i], streams[i]);
        }

        // Wait and collect per-stream times (wall = max across streams)
        float kernelWallMs = 0.0f;
        float copyWallMs = 0.0f;
        for (int i = 0; i < NUM_STREAMS; i++) {
            cuStreamSynchronize(streams[i]);
            float[] k = new float[1];
            float[] c = new float[1];
            cuEventElapsedTime(k, evKernelStart[i], evKernelEnd[i]);
            cuEventElapsedTime(c, evKernelEnd[i], evCopyEnd[i]);
            kernelWallMs = Math.max(kernelWallMs, k[0]);
            copyWallMs = Math.max(copyWallMs, c[0]);
        }

        // Optional: write full results file
        long tBeforeWrite = System.nanoTime();
        if (writeResults) {
            try (ResultsWriter writer = new ResultsWriter("fourier_results.bin", LENGTH)) {
                for (int i = 0; i < NUM_STREAMS; i++) {
                    int startIdx = i * CHUNK_SIZE;
                    int cur = currentSizes[i];
                    if (cur == 0) continue;
                    // Use a view limited to 'cur' elements
                    FloatBuffer view = hBuf[i].duplicate();
                    view.limit(cur);
                    writer.writeChunk(startIdx, view, cur);
                }
            }
        }
        long tBeforeCleanup = System.nanoTime();

        // Cleanup
        for (int i = 0; i < NUM_STREAMS; i++) {
            cuMemFree(dOut[i]);
            cuMemFreeHost(hPinned[i]);
            cuStreamDestroy(streams[i]);
            cuEventDestroy(evKernelStart[i]);
            cuEventDestroy(evKernelEnd[i]);
            cuEventDestroy(evCopyEnd[i]);
        }
        long tEnd = System.nanoTime();

        double prepMs = (tPrepEnd - t0) / 1e6;
        double deallocMs = (tEnd - tBeforeCleanup) / 1e6;
        double totalMs = (tEnd - t0) / 1e6;
        return new double[] { prepMs, kernelWallMs, copyWallMs, deallocMs, totalMs };
    }

    // ===== Utilities =====
    private static int gridFor(int n) {
        if (n <= 0) return 1;
        return (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    }

    /** Simple statistics (mean & stddev). */
    private static final class Stats {
        private final double[] v;
        private int i = 0;
        Stats(int n) { v = new double[n]; }
        void add(double ms) { v[i++] = ms; }
        double mean() {
            double s = 0.0;
            for (double x : v) s += x;
            return s / v.length;
        }
        double std() {
            double m = mean(), s = 0.0;
            for (double x : v) { double d = x - m; s += d * d; }
            return Math.sqrt(s / Math.max(1, v.length - 1));
        }
    }

    /** Driver-API handles to __constant__ symbols. */
    private static final class ConstantSymbols {
        private final CUdeviceptr tminPtr = new CUdeviceptr();
        private final CUdeviceptr deltaPtr = new CUdeviceptr();
        private final CUdeviceptr coeffPtr = new CUdeviceptr();
        private final CUdeviceptr piPtr = new CUdeviceptr();
        private final CUdeviceptr pi2Ptr = new CUdeviceptr();
        private final CUdeviceptr Tptr = new CUdeviceptr();
        private final CUdeviceptr piOverTPtr = new CUdeviceptr();
        private final CUdeviceptr resultCoeffPtr = new CUdeviceptr();

        private ConstantSymbols() {}

        static ConstantSymbols resolve(CUmodule module) {
            ConstantSymbols cs = new ConstantSymbols();
            long[] sz = new long[1];
            cuModuleGetGlobal(cs.tminPtr, sz, module, "const_tmin");
            cuModuleGetGlobal(cs.deltaPtr, sz, module, "const_delta");
            cuModuleGetGlobal(cs.coeffPtr, sz, module, "const_coefficients");
            cuModuleGetGlobal(cs.piPtr, sz, module, "const_pi");
            cuModuleGetGlobal(cs.pi2Ptr, sz, module, "const_pi_squared");
            cuModuleGetGlobal(cs.Tptr, sz, module, "const_T");
            cuModuleGetGlobal(cs.piOverTPtr, sz, module, "const_pi_over_T");
            cuModuleGetGlobal(cs.resultCoeffPtr, sz, module, "constant_result_coefficient");
            return cs;
        }

        void uploadStatic() {
            cuMemcpyHtoD(deltaPtr, Pointer.to(new float[]{ DELTA }), Sizeof.FLOAT);
            cuMemcpyHtoD(coeffPtr, Pointer.to(new int[]{ COEFFICIENTS }), Sizeof.INT);
            cuMemcpyHtoD(piPtr, Pointer.to(new float[]{ PI }), Sizeof.FLOAT);
            cuMemcpyHtoD(pi2Ptr, Pointer.to(new float[]{ PI_SQUARED }), Sizeof.FLOAT);
            cuMemcpyHtoD(Tptr, Pointer.to(new float[]{ PERIOD }), Sizeof.FLOAT);
            cuMemcpyHtoD(piOverTPtr, Pointer.to(new float[]{ PI_OVER_T }), Sizeof.FLOAT);
            cuMemcpyHtoD(resultCoeffPtr, Pointer.to(new float[]{ RESULT_COEFFICIENT }), Sizeof.FLOAT);
        }

        void setTmin(float tmin) {
            cuMemcpyHtoD(tminPtr, Pointer.to(new float[]{ tmin }), Sizeof.FLOAT);
        }
    }

    /** Writes results into a single binary file of LENGTH floats. */
    private static final class ResultsWriter implements AutoCloseable {
        private final RandomAccessFile raf;
        private final FileChannel ch;

        ResultsWriter(String path, long totalFloats) {
            try {
                raf = new RandomAccessFile(path, "rw");
                ch = raf.getChannel();
                raf.setLength(totalFloats * Sizeof.FLOAT);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        void writeChunk(int startIdx, FloatBuffer buf, int count) {
            try {
                long bytePos = (long) startIdx * Sizeof.FLOAT;
                ByteBuffer bb = ByteBuffer.allocateDirect(count * Sizeof.FLOAT)
                        .order(ByteOrder.nativeOrder());
                buf.rewind();
                for (int i = 0; i < count; i++) {
                    bb.putFloat(buf.get());
                }
                bb.flip();
                ch.position(bytePos);
                while (bb.hasRemaining()) {
                    ch.write(bb);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void close() {
            try {
                ch.close();
                raf.close();
            } catch (IOException ignored) {
            }
        }
    }
}
