package org.jcuda.kacygan.mastersdeg;

public interface FourierTest {
    String FUNCTION_NAME = "fourier";
    int LENGTH = 1_000_000_000;
    int COEFFICIENTS = 1024;
    float TMIN = -3.0f;
    float TMAX = 3.0f;
    int THREADS_PER_BLOCK = 256;
    double THOUSAND = 1000.0;
    float DELTA = (TMAX - TMIN) / (LENGTH - 1);
    float PERIOD = 1.0f;
    float PI = 3.14159265f;
    float PI_OVER_T = (PI / PERIOD);
    float PI_SQUARED = PI * PI;
    float RESULT_COEFFICIENT = (4.0f) / (PI_SQUARED);
    int NUM_STREAMS = 4;
    boolean logReps = false;

    int N = 10;
    int REPS = 10;
    int NUM_REPS = 20;

    void runTest();
}
