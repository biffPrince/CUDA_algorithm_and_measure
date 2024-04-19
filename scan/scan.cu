#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <numeric>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
__global__ void scan(int *d_a, int N)
{

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int stride = 1; stride < N; stride <<= 1)
    {
        if (idx >= stride)
            d_a[idx] += d_a[idx - stride];
        __syncthreads(); //only sync blocks. here is data race.
    }
}

// scan for MIT algorithm. https://people.csail.mit.edu/xchen/gpu-programming/Lecture08-scan.pdf 
// work for multi block.
__global__ void scan2(int *g_data, int n)
{
    cg::thread_block cta = cg::this_thread_block();
    //reduce for stride 1,2,4.
    // round 1 a1+=a0, a3+=a2,a5+=a4,a7+=a6;round 2 a3+=a1,a7+=a6; round 3 a7+=a3;
    // round 1 idx+1 mod 2==0; round 2 idx+1 mod 4==0; round 3 idx+1 mod 8==0; 
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    for(int stride = 1; stride < n; stride <<= 1)
    {
        if((idx+1)%(stride*2)==0 &&idx<n)
            g_data[idx] += g_data[idx-stride];
        cg::sync(cta);
    }
    // reverse reduce
    for(int stride = n>>2; stride > 0; stride >>= 1)
    {
        // n=8, stride=2, idx=0, 1, 2, 3, 4, 5, 6, 7; index=3,7 ;index need < n-1 a[5] added;
        // stride=1,idx=0,1,2,3,4,5,6,7; index=1,3,5,7...; a[2]
        // round1 a5+=a3,stride=2,idx=5, S5=a0+...+a5; round 2: a2+=a1, a4+=a3, a6+=a5
        // 16: a11+=a7; 
        if((idx+1)%(stride*2)==0 && (idx+1)>=stride*2 && idx<n)
            g_data[idx+stride]+=g_data[idx];
        cg::sync(cta);
    }
}

int main()
{
    // generate two arrays, one host, one device.
    int N = 1 << 12;
    int *h_a = new int[N];
    int *h_test = new int[N];
    int *h_b = new int[N]; // this is device return value. to check if the scan is correct.
    int *d_a;
    int *d_b;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    // fill the arrays with 1.
    std::fill(h_a, h_a + N, 1);
    // fill test with 1,2,3,4...
    std::iota(h_test, h_test + N, 1);
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    // scan on host.
    std::partial_sum(h_a, h_a + N, h_a);

    std::cout << std::endl;
    // test h_a with h_test
    if (std::equal(h_a, h_a + N, h_test))
        std::cout << "scan is correct" << std::endl;
    else
        std::cout << "scan is wrong" << std::endl;
    // scan on device.
    // scan<<<1, N>>>(d_a, N);
    // cudaMemcpy(h_b, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);
    scan2<<<4, 1024>>>(d_a, N);
    cudaMemcpy(h_b, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);
    // test device return value equal to host value, use stl
    if (std::equal(h_a, h_a + N, h_b))
        std::cout << "device scan is correct" << std::endl;
    else
        std::cout << "device scan is wrong" << std::endl;

    // free all the memory.
    delete[] h_a;
    delete[] h_b;
    delete[] h_test;
    cudaFree(d_b);
    cudaFree(d_a);
    return 0;
}