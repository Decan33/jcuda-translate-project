#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cstring>
#include "data.h"

constexpr int BATCH_SIZE = 16 * 1024 * 1024;

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void fourier(
    float tmin,
    float delta,
    int length,
    int coefficients,
    float pi,
    float pi_over_T,
    float result_coefficient,
    float T,
    float *results,
    int stream_offset,
    int stream_size)
{
    auto idx_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_local >= stream_size) return;
    auto idx_global = idx_local + stream_offset;
    auto t = tmin + idx_global * delta;
    auto sum = 0.0f;
    for (auto k = 1; k <= coefficients; ++k) {
        auto angle = (2 * k - 1) * pi_over_T * t;
        auto denominator = 4.0f * k * k - 4.0f * k + 1.0f;
        sum += cosf(angle) / denominator;
    }
    results[idx_local] = T * 0.5f - result_coefficient * sum;
}

void performColdRun(float tmin, float tmax, int length, int coefficients, float T, 
                   float delta, float pi, float pi_over_T, float result_coefficient) {
    cudaStream_t streams[NUM_STREAMS];
    for (auto i = 0; i < NUM_STREAMS; ++i) {
        HANDLE_ERROR(cudaStreamCreate(&streams[i]));
    }
    float *d_results[NUM_STREAMS][2];
    float *h_results[NUM_STREAMS][2];
    for (auto i = 0; i < NUM_STREAMS; ++i) {
        for (auto b = 0; b < 2; ++b) {
            HANDLE_ERROR(cudaMalloc(&d_results[i][b], BATCH_SIZE * sizeof(float)));
            HANDLE_ERROR(cudaMallocHost(&h_results[i][b], BATCH_SIZE * sizeof(float)));
        }
    }
    auto final_results = new float[length];

    auto threadsPerBlock = THREADS_PER_BLOCK;
    auto num_batches = (length + (BATCH_SIZE * NUM_STREAMS) - 1) / (BATCH_SIZE * NUM_STREAMS);
    for (auto batch = 0; batch < num_batches; ++batch) {
        for (auto s = 0; s < NUM_STREAMS; ++s) {
            auto global_offset = (batch * NUM_STREAMS + s) * BATCH_SIZE;
            if (global_offset >= length) continue;
            auto current_batch_size = (global_offset + BATCH_SIZE > length) ? (length - global_offset) : BATCH_SIZE;
            auto buf = batch % 2;
            auto blocksPerGrid = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;
            fourier<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
                tmin, delta, length, coefficients, pi, pi_over_T, result_coefficient, T,
                d_results[s][buf], global_offset, current_batch_size
            );
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaMemcpyAsync(
                h_results[s][buf], d_results[s][buf],
                current_batch_size * sizeof(float), cudaMemcpyDeviceToHost, streams[s]
            ));
        }
        if (batch > 0) {
            auto prev_buf = (batch - 1) % 2;
            for (auto s = 0; s < NUM_STREAMS; ++s) {
                auto prev_offset = ((batch - 1) * NUM_STREAMS + s) * BATCH_SIZE;
                if (prev_offset >= length) continue;
                auto prev_batch_size = (prev_offset + BATCH_SIZE > length) ? (length - prev_offset) : BATCH_SIZE;
                HANDLE_ERROR(cudaStreamSynchronize(streams[s]));
                std::memcpy(final_results + prev_offset, h_results[s][prev_buf], prev_batch_size * sizeof(float));
            }
        }
    }
    auto last_batch = num_batches - 1;
    auto last_buf = last_batch % 2;
    for (auto s = 0; s < NUM_STREAMS; ++s) {
        auto last_offset = (last_batch * NUM_STREAMS + s) * BATCH_SIZE;
        if (last_offset >= length) continue;
        auto last_batch_size = (last_offset + BATCH_SIZE > length) ? (length - last_offset) : BATCH_SIZE;
        HANDLE_ERROR(cudaStreamSynchronize(streams[s]));
        std::memcpy(final_results + last_offset, h_results[s][last_buf], last_batch_size * sizeof(float));
    }

    // Cleanup
    for (auto i = 0; i < NUM_STREAMS; ++i) {
        for (auto b = 0; b < 2; ++b) {
            cudaFree(d_results[i][b]);
            cudaFreeHost(h_results[i][b]);
        }
    }
    delete[] final_results;
    for (auto i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    constexpr float tmin = -3.0f;
    constexpr float tmax = 3.0f;
    constexpr int length = 200000000;
    constexpr int coefficients = 1024;
    constexpr float T = 1.0f;
    constexpr float delta = (tmax - tmin) / (length - 1);
    constexpr float pi = 3.14159265f;
    constexpr float pi_sq = pi * pi;
    constexpr float pi_over_T = pi / T;
    constexpr float result_coefficient = (4.0f * T) / pi_sq;

    printf("Performing cold run to warm up GPU...\n");
    performColdRun(tmin, tmax, length, coefficients, T, delta, pi, pi_over_T, result_coefficient);
    printf("Cold run completed.\n\n");

    std::vector<double> prep_times, kernel_times, delete_times;

    auto start_reps = std::chrono::high_resolution_clock::now();
    for (auto rep = 0; rep < NUM_REPS; ++rep) {
        auto prep_start = std::chrono::high_resolution_clock::now();
        
        cudaStream_t streams[NUM_STREAMS];
        for (auto i = 0; i < NUM_STREAMS; ++i) {
            HANDLE_ERROR(cudaStreamCreate(&streams[i]));
        }
        
        float *d_results[NUM_STREAMS][2];
        float *h_results[NUM_STREAMS][2];
        for (auto i = 0; i < NUM_STREAMS; ++i) {
            for (auto b = 0; b < 2; ++b) {
                HANDLE_ERROR(cudaMalloc(&d_results[i][b], BATCH_SIZE * sizeof(float)));
                HANDLE_ERROR(cudaMallocHost(&h_results[i][b], BATCH_SIZE * sizeof(float)));
            }
        }
        
        auto final_results = new float[length];
        
        auto prep_end = std::chrono::high_resolution_clock::now();
        prep_times.push_back(std::chrono::duration<double>(prep_end - prep_start).count());

        cudaEvent_t kernel_start, kernel_stop;
        HANDLE_ERROR(cudaEventCreate(&kernel_start));
        HANDLE_ERROR(cudaEventCreate(&kernel_stop));
        HANDLE_ERROR(cudaEventRecord(kernel_start));

        auto threadsPerBlock = THREADS_PER_BLOCK;
        auto num_batches = (length + (BATCH_SIZE * NUM_STREAMS) - 1) / (BATCH_SIZE * NUM_STREAMS);
        
        for (auto batch = 0; batch < num_batches; ++batch) {
            for (auto s = 0; s < NUM_STREAMS; ++s) {
                auto global_offset = (batch * NUM_STREAMS + s) * BATCH_SIZE;
                if (global_offset >= length) continue;
                
                auto current_batch_size = (global_offset + BATCH_SIZE > length) ? (length - global_offset) : BATCH_SIZE;
                auto buf = batch % 2;
                auto blocksPerGrid = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;
                
                fourier<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
                    tmin, delta, length, coefficients, pi, pi_over_T, result_coefficient, T,
                    d_results[s][buf], global_offset, current_batch_size
                );
                HANDLE_ERROR(cudaGetLastError());
                HANDLE_ERROR(cudaMemcpyAsync(
                    h_results[s][buf], d_results[s][buf],
                    current_batch_size * sizeof(float), cudaMemcpyDeviceToHost, streams[s]
                ));
            }
            
            if (batch > 0) {
                auto prev_buf = (batch - 1) % 2;
                for (auto s = 0; s < NUM_STREAMS; ++s) {
                    auto prev_offset = ((batch - 1) * NUM_STREAMS + s) * BATCH_SIZE;
                    if (prev_offset >= length) continue;
                    
                    auto prev_batch_size = (prev_offset + BATCH_SIZE > length) ? (length - prev_offset) : BATCH_SIZE;
                    HANDLE_ERROR(cudaStreamSynchronize(streams[s]));
                    std::memcpy(final_results + prev_offset, h_results[s][prev_buf], prev_batch_size * sizeof(float));
                }
            }
        }
        
        auto last_batch = num_batches - 1;
        auto last_buf = last_batch % 2;
        for (auto s = 0; s < NUM_STREAMS; ++s) {
            auto last_offset = (last_batch * NUM_STREAMS + s) * BATCH_SIZE;
            if (last_offset >= length) continue;
            
            auto last_batch_size = (last_offset + BATCH_SIZE > length) ? (length - last_offset) : BATCH_SIZE;
            HANDLE_ERROR(cudaStreamSynchronize(streams[s]));
            std::memcpy(final_results + last_offset, h_results[s][last_buf], last_batch_size * sizeof(float));
        }
        
        HANDLE_ERROR(cudaEventRecord(kernel_stop));
        HANDLE_ERROR(cudaEventSynchronize(kernel_stop));
        
        float kernel_ms = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms / 1000.0);
        
        HANDLE_ERROR(cudaEventDestroy(kernel_start));
        HANDLE_ERROR(cudaEventDestroy(kernel_stop));

        auto delete_start = std::chrono::high_resolution_clock::now();
        
        for (auto i = 0; i < NUM_STREAMS; ++i) {
            for (auto b = 0; b < 2; ++b) {
                cudaFree(d_results[i][b]);
                cudaFreeHost(h_results[i][b]);
            }
        }
        
        delete[] final_results;
        for (auto i = 0; i < NUM_STREAMS; ++i) {
            cudaStreamDestroy(streams[i]);
        }
        
        auto delete_end = std::chrono::high_resolution_clock::now();
        delete_times.push_back(std::chrono::duration<double>(delete_end - delete_start).count());
    }
    auto end_reps = std::chrono::high_resolution_clock::now();
    
    double prep_sum = 0, kernel_sum = 0, delete_sum = 0;
    printf("\n===== Timing Summary =====\n");
    
    for (auto i = 0u; i < prep_times.size(); ++i) {
        printf("Repetition %u:\n", i + 1);
        printf("  Preparation time: %.6f s\n", prep_times[i]);
        printf("  Kernel execution time: %.6f s\n", kernel_times[i]);
        printf("  Memory deletion time: %.6f s\n", delete_times[i]);
        prep_sum += prep_times[i];
        kernel_sum += kernel_times[i];
        delete_sum += delete_times[i];
    }
    
    auto n = static_cast<double>(prep_times.size());
    double prep_avg = prep_sum / n;
    double kernel_avg = kernel_sum / n;
    double delete_avg = delete_sum / n;
    double prep_var = 0, kernel_var = 0, delete_var = 0;
    
    for (auto i = 0u; i < prep_times.size(); ++i) {
        prep_var += (prep_times[i] - prep_avg) * (prep_times[i] - prep_avg);
        kernel_var += (kernel_times[i] - kernel_avg) * (kernel_times[i] - kernel_avg);
        delete_var += (delete_times[i] - delete_avg) * (delete_times[i] - delete_avg);
    }
    
    double prep_std = std::sqrt(prep_var / n);
    double kernel_std = std::sqrt(kernel_var / n);
    double delete_std = std::sqrt(delete_var / n);
    
    printf("\nAverages over %zu repetitions:\n", prep_times.size());
    printf("  Avg preparation time: %.6f s (stddev: %.6f s)\n", prep_avg, prep_std);
    printf("  Avg kernel execution time: %.6f s (stddev: %.6f s)\n", kernel_avg, kernel_std);
    printf("  Avg memory deletion time: %.6f s (stddev: %.6f s)\n", delete_avg, delete_std);
    printf("  Whole time taken for %d reps: %.6f s\n", NUM_REPS, std::chrono::duration<double>(end_reps - start_reps).count());
    printf("=========================\n\n");
    return 0;
}
