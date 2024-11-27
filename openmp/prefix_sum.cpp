#include "vectorio.h" // Include the header for file operations
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <fstream>

// Prefix sum implementation
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
    // File paths for input and output
    std::string input_file = "input_10k.bin";
    std::string output_file = "output_10k.bin";

    // Read input and expected output vectors from files
    std::vector<int> A = read_from_file(input_file);
    std::vector<int> expected_output = read_from_file(output_file);

    if (A.empty()) {
        std::cerr << "Error: Input file is empty or not properly formatted." << std::endl;
        return 1;
    }

    // Prepare output vector
    std::vector<int> B(A.size());

    // Perform prefix sum
    int num_threads = 32; // Example: Use 32 threads
    auto start = std::chrono::high_resolution_clock::now();
    prefix_sum(A, B, num_threads);
    auto end = std::chrono::high_resolution_clock::now();

    // Measure execution time
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution Time: " << duration.count() << " seconds" << std::endl;

    // Write the result to output file
    write_to_file("prefix_sum_output.bin", B);
    std::cout << "Output written to prefix_sum_output.bin" << std::endl;

    // Optional: Verify against expected output if available
    if (!expected_output.empty()) {
        if (expected_output.size() != B.size()) {
            std::cerr << "Error: Expected output size does not match result size." << std::endl;
            return 1;
        }

        bool correct = true;
        for (size_t i = 0; i < B.size(); ++i) {
            if (B[i] != expected_output[i]) {
                std::cerr << "Mismatch at index " << i 
                          << ": Computed " << B[i] 
                          << ", Expected " << expected_output[i] << std::endl;
                correct = false;
                break;
            }
        }

        if (correct) {
            std::cout << "Result matches expected output." << std::endl;
        } else {
            std::cerr << "Result does not match expected output." << std::endl;
        }
    }

    return 0;
}