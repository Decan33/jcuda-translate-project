#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

constexpr int NUM_REPS = 20;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_COEFFICIENTS = 1024;
constexpr float tmin = -3.0f;
constexpr float tmax = 3.0f;
constexpr int length = 1000000000;
constexpr int coefficients = 1024;
constexpr float T = 1.0f;
constexpr float delta = (tmax - tmin) / (length - 1);
constexpr float pi = 3.14159265f;
constexpr float pi_sq = pi * pi;
constexpr float pi_over_T = pi / T;
constexpr float pi_squared = pi * pi;
constexpr float result_coefficient = (4.0f * T) / pi_sq;
constexpr int NUM_STREAMS = 4;
constexpr bool logReps = 0;

const size_t CHUNK_ELEMS = 1u << 20;
const size_t CHUNK_BYTES = CHUNK_ELEMS * sizeof(float);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)