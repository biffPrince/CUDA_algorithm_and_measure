#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__global__ void histogram(float *d_data, int N, float *d_hist, int num_bins, float bin_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        int bin = d_data[idx] / bin_width;
        atomicAdd(&d_hist[bin], 1.0f);
    }
}

int main()
{
    // Set up the random number generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Generate random numbers
    const int N = 100'0000;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    curandGenerateUniform(gen, d_data, N);

    // Set up the histogram
    const int num_bins = 32;
    float bin_width = 1.0f / num_bins;
    float *d_hist;
    cudaMalloc(&d_hist, num_bins * sizeof(float));
    cudaMemset(d_hist, 0, num_bins * sizeof(float));

    // Compute the histogram
    histogram<<<(N + 255) / 256, 256>>>(d_data, N, d_hist, num_bins, bin_width);

    // Copy the histogram back to the host
    float *h_hist = new float[num_bins];
    cudaMemcpy(h_hist, d_hist, num_bins * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the histogram
    for (int i = 0; i < num_bins; i++)
    {
        std::cout << h_hist[i] << std::endl;
    }

    //create a vector to store the histogram
    std::vector<float> hist(h_hist, h_hist + num_bins);
    //find the maximum element in the histogram
    auto max = std::max_element(hist.begin(), hist.end());
    //find the index of the maximum element
    auto index = std::distance(hist.begin(), max);
    //print the index of the maximum element
    std::cout << "The index of the maximum element is: " << index << std::endl;
    //use cpu test algorithm run time
    std::vector<float> data(N);
    //random generate data use <random> library and <chrono> library to calculate the time
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    std::generate(data.begin(), data.end(), [&]() { return distribution(generator); });
    // Clean up
    delete[] h_hist;
    cudaFree(d_hist);
    cudaFree(d_data);
    curandDestroyGenerator(gen);
    return 0;
}