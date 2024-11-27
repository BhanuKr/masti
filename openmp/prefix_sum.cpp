#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

void prefix_sum(const std::vector<int>& A, std::vector<int>& B, int num_threads) {
    int n = A.size();
    B[0] = A[0];

    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int start = id * n / nth;
        int end = (id + 1) * n / nth;

        for (int i = start + 1; i < end; ++i) {
            B[i] = B[i - 1] + A[i];
        }

        #pragma omp barrier

        int offset = 0;
        for (int i = 0; i < id; ++i) {
            offset += B[(i + 1) * n / nth - 1];
        }

        for (int i = start; i < end; ++i) {
            B[i] += offset;
        }
    }
}

int main() {
    std::vector<int> sizes = {10000, 20000, 30000, 1000000};
    std::vector<int> threads = {2, 4, 8, 16, 32, 64};

    for (int size : sizes) {
        std::vector<int> A(size);
        std::iota(A.begin(), A.end(), 1);
        std::vector<int> B(size);

        // Measure sequential execution time
        double sequential_time = 0.0;
        for (int run = 0; run < 5; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            prefix_sum(A, B, 1);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            sequential_time += duration.count();
        }
        sequential_time /= 5.0;

        for (int num_threads : threads) {
            double total_time = 0.0;

            for (int run = 0; run < 5; ++run) {
                auto start = std::chrono::high_resolution_clock::now();
                prefix_sum(A, B, num_threads);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                total_time += duration.count();
            }

            double average_time = total_time / 5.0;
            double speedup = sequential_time / average_time;
            std::cout << "Size: " << size << ", Threads: " << num_threads << ", Time: " << average_time << " seconds, Speedup: " << speedup << std::endl;
        }
    }

    return 0;
}