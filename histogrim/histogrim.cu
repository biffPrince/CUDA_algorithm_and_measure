/*
    * histogrim.cu
    * test for histogrim speed.
*/
#include <iostream>
#include <random>
#include <algorithm>
#include <map>
#include <numeric>
#include <chrono>

#include <cuda_runtime.h>

// static auto dice()
// {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0,31);
//     return dis(gen);
// }

__global__ void histogrim(int n,int *a,unsigned int *b)
{
    // __shared__ unsigned int hist[32];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n )
    {
        atomicAdd(&b[a[index]],1);
    }

}

__global__ void histogrim_shared_memory(int n,int *a,unsigned int *b)
{
    __shared__ unsigned int hist[32];
    if(threadIdx.x<32)
    {
        hist[threadIdx.x]=0;
    }
    __syncthreads();
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n )
    {
        atomicAdd(&hist[a[index]],1);
    }
    __syncthreads();
    if(threadIdx.x<32)
    {
        atomicAdd(&b[threadIdx.x],hist[threadIdx.x]);
    }
}
struct dice {
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis;

    dice() : gen(rd()), dis(0, 31) {}

    int operator()() {
        return dis(gen);
    }
};
int main()
{
    cudaFree(0);
    // allocate memory on host
    const int n = 1 << 24; 
    int *a = new int[n];
    //create random generate engine to generate [0,31]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 31);
    std::generate(a, a + n, [&]() { return dis(gen); });

    //calculate it in host.
    std::map<int,int> hist;
    //measure CPU calculation time
    auto start = std::chrono::high_resolution_clock::now(); 
    std::for_each(a,a+n,[&](int i){
        hist[i]++;
    });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The elapsed time of CPU in microseconds is " << duration.count() << std::endl;

    //create device memory
    int *d_a;
    cudaMalloc(&d_a, n*sizeof(int));
    //copy data from host to device
    cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
    //create histogrim memory
    unsigned int *d_b;
    cudaMalloc(&d_b, 32*sizeof(unsigned int));
    //launch kernel
    // histogrim<<<(n+255)/256,256>>>(n,d_a,d_b);
    histogrim_shared_memory<<<(n+255)/256,256>>>(n,d_a,d_b);
    //copy data from device to host
    unsigned int *b = new unsigned int[32];
    cudaMemcpy(b, d_b, 32*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //compare the result
    bool state=false;
    for(int i=0;i<32;i++)
    {
        if(hist[i]!=b[i])
        {
            state=true;
            break;
        }
    }
    if(state)
    {
        std::cout<<"The histogrim results in device and host are not equal"<<std::endl;
    }else
    {
        std::cout<<"The histogrim results in device and host are equal"<<std::endl;
    }
    //free resource
    delete[] a;
    delete[] b;
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}