#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
// curand in device API
__global__ void init(unsigned int seed, curandState_t* states) 
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}
// use XORWOW algorithm to generate random number  with uniform distribution.
__global__ void generate(curandState_t* states, float* numbers) 
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    numbers[id] = curand_uniform(&states[id]);
}


int main()
{
    float *d_A;
    const int size= 1000'0000;
    // malloc device memory
    cudaMalloc(&d_A,size*sizeof(float));
    
    //use cuda_runtime measure curand in host API;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); //start record time
    //generate random number
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,2233ULL); //set seed
    curandGenerateUniform(gen,d_A,size);
    cudaEventRecord(stop); //stop record time
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime,start,stop);
    printf("Time to generate random number in host API: %3.1f ms\n",elapsedTime);

    //use cuda_runtime measure curand in device API;
    curandState_t* states;
    cudaMalloc(&states,size*sizeof(curandState_t));
    cudaEventRecord(start); //start record time
    init<<<size/256,256>>>(2237,states);
    generate<<<size/256,256>>>(states,d_A);
    cudaEventRecord(stop); //stop record time
    cudaEventSynchronize(stop);
    float elapsedTimeDevice;
    cudaEventElapsedTime(&elapsedTimeDevice,start,stop);
    printf("Time to generate random number in device API: %3.1f ms\n",elapsedTimeDevice);

    //free memory
    cudaFree(d_A);
    cudaFree(states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    curandDestroyGenerator(gen);
    return 0;
}