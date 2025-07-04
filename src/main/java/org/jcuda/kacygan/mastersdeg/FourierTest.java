package org.jcuda.kacygan.mastersdeg;

public interface FourierTest {
    String FUNCTION_NAME = "fourier";
    int NUM_REPS = 20;
    int LENGTH = 1_000_000_000;
    int COEFFICIENTS = 1024;
    float TMIN = -3.0f;
    float TMAX = 3.0f;
    int THREADS_PER_BLOCK = 256;
    double THOUSAND = 1000.0;

    void runTest();
}
