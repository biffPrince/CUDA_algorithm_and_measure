#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

int main() {
    // Create a random number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    // Create two vectors of size 10,000,000
    std::vector<float> a(1000'0000);
    std::vector<float> b(1000'0000);
    std::vector<float> c(1000'0000);

    // Fill the vectors with random numbers
    auto begin = std::chrono::high_resolution_clock::now();
    std::generate(a.begin(), a.end(), [&]() { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&]() { return distribution(generator); });
    auto end = std::chrono::high_resolution_clock::now();
    auto durationGen = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Time taken to fill vectors: " << durationGen.count() << " microseconds" << std::endl;
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    // Add the vectors using std::transform
    std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<float>());

    // Stop the timer
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    return 0;
}