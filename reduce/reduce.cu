#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>

namespace cg = cooperative_groups;

//reduction use kernel,warps not active, inefficient.
__global__ void reduce0(int* g_idata,int n, int* g_odata) {
    // cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = i< n ? g_idata[i] : 0;
    // cg::sync(cta);
    __syncthreads();// cg better than __syncthreads().
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        // cg::sync(cta);
        __syncthreads();//namespace cg = cooperative_groups;;
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
// replace % with * to improve the efficiency, avoid bank conflict.
__global__ void reduce1(int* g_idata, int n, int* g_odata) {
    // cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = i < n ? g_idata[i] : 0;
    // cg::sync(cta);
    __syncthreads(); //cg better than __syncthreads().
    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s >0; s >>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // cg::sync(cta);
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// use the warp shuffle instruction to improve the efficiency.
//this is slow, because GPU not good at function call. and has small stack size.
__device__ void warpReduce(volatile int* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
__global__ void reduce2(int* g_idata, int n, int* g_odata) {
    // cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = i < n ? g_idata[i] : 0;
    // cg::sync(cta);
    __syncthreads();  //cg better than __syncthreads().
    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s >32; s >>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // cg::sync(cta);
        __syncthreads();
    }
    if(tid<32)
        warpReduce(sdata,tid);
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// use the warp shuffle instruction to improve the efficiency.
__global__ void reduce3(int* g_idata, int n, int* g_odata) {
    // cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = i < n ? g_idata[i] : 0;
    // cg::sync(cta);
    __syncthreads(); //cg better than __syncthreads().
    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s >=32; s >>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // cg::sync(cta);
        __syncthreads();
    }
    //use warp shuffle to reduce the warp size to 32.
    if(tid<32)
    {
            // volatile int* smem = sdata;
            // smem[tid] += smem[tid + 32];
            // smem[tid] += smem[tid + 16];
            // smem[tid] += smem[tid + 8];
            // smem[tid] += smem[tid + 4];
            // smem[tid] += smem[tid + 2];
            // smem[tid] += smem[tid + 1];
        // if block size bigger than 64 
        for(int offset=32/2;offset>0;offset >>=1)
        {
            // sdata[tid]+=__shfl_down(sdata[tid],offset);
            sdata[tid] += __shfl_down_sync(0xffffffff, sdata[tid], offset);
        }
    }
    // __syncwarp();
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main()
{
    // set array in CPU and GPU
    const int N = 1 << 20;
    int* d_data, * h_data;
    h_data = new int[N];
    cudaMalloc(&d_data, N * sizeof(int));
    // generate random numbers
    // std::default_random_engine generator;
    // std::uniform_real_distribution<float> distribution(0.0, 1.0);
    // for (int i = 0; i < N; i++)
    // {
    //     h_data[i] = distribution(generator);
    // }
    // generate all 1.0 to array with stl
    std::fill(h_data, h_data + N, 1);

    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    // set up the reduction
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    int* d_result;
    cudaMalloc(&d_result, num_blocks*sizeof(int));
    // compute the reduction
    reduce3 << <num_blocks, block_size,block_size*sizeof(int) >> > (d_data,N,d_result);
    // copy the result back to the host
    int hresult0[num_blocks];
    cudaMemcpy(&hresult0, d_result, num_blocks* sizeof(int), cudaMemcpyDeviceToHost);
    // print the result
    auto G_result = std::accumulate(hresult0, hresult0+num_blocks, 0);
    std::cout << "device compute: " << G_result << std::endl;
    // use CPU to test the algorithm run time
    auto result = std::accumulate(h_data, h_data + N, 0);
    
    std::cout << "CPU compute: " << result << std::endl;

    if (result - G_result == 0)
        std::cout << "Passed reduce" << std::endl;

    // free the memory
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_result);
    return 0;
}