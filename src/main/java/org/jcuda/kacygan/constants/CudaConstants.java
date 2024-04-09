package org.jcuda.kacygan.constants;

public enum CudaConstants {
    CHAPTER_10_N(1024*1024),
    CHAPTER_10_FULL_DATA_SIZE(CHAPTER_10_N.size * 20),
    CHAPTER_10_COPY_TIMED_TEST(64* CHAPTER_10_N.size)
    ;


    public final int size;

    CudaConstants(int i) {
        size = i;
    }
}
