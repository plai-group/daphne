#include <curand_kernel.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

using namespace std;

// Helper function to check CUDA errors
void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

// Kernel to initialize random states
__global__ void init_random_states(curandState* states, int seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_samples>" << std::endl;
        return 1;
    }
    int N = std::stoi(argv[1]);  // Number of samples
    int K = %d; // std::stoi(argv[2]);  // Number of sample statements
    int L = %d; // std::stoi(argv[3]);  // Number of observation statements

    // Allocate device memory for samples, observations, and results
    float* samples;
    float* log_probs;
    curandState* rand_states;
    checkCudaError(cudaMalloc((void**)&samples, N * K * sizeof(float)));
    checkCudaError(cudaMalloc((void**)&log_probs, N * L * sizeof(float)));  // Six observations per sample pair
    checkCudaError(cudaMalloc((void**)&rand_states, N * sizeof(curandState)));

    // Initialize random states
    int seed = 0;
    init_random_states<<<(N + 255) / 256, 256>>>(rand_states, seed);
    cudaDeviceSynchronize();

    // Launch the combined sampling and observation kernel
    generate_samples<<<(N + 255) / 256, 256>>>(rand_states, samples, log_probs, N, K);
    cudaDeviceSynchronize();

    // Copy results back to host
    float* res_samples = new float[N * K];
    float* res_log_probs = new float[N * L];
    checkCudaError(cudaMemcpy(res_samples, samples, N * K * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(res_log_probs, log_probs, N * L * sizeof(float), cudaMemcpyDeviceToHost));

    // Print samples and log_probs
    for (int i = 0; i < N; ++i) {
        std::cout << "Sample " << i << ": ";
        for (int j = 0; j < K; ++j) {
            std::cout << res_samples[i * K + j] << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < N; ++i) {
        std::cout << "Observe " << i << ": ";
        for (int j = 0; j < L; ++j) {
            std::cout << res_log_probs[i * L + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    checkCudaError(cudaFree(samples));
    checkCudaError(cudaFree(log_probs));
    checkCudaError(cudaFree(rand_states));
    delete[] res_samples;
    delete[] res_log_probs;

    return 0;
}