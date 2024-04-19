#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>

// it can be optimaized with grid-stride loop.
__global__ void vectorAdd(int n, float *x, float *y,float *z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        // z[i] =  x[i] + y[i];
        z[i]=__fadd_rn(x[i],y[i]);
}

__global__ void compareVectorsKernel(float* d, float* b, int* areEqual, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n && (fabs(d[index] - b[index]) > 1e-6)){
        *areEqual = 0;
        return ;
    }
}

int main()
{
    const int n = 1000'0000;
    //all is device memory
    float *a, *b,*d ;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&d, n * sizeof(float));
    //use kernel function and measure it
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Set up the random number generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123ULL);
    curandGenerateUniform(gen,a,n);
    curandGenerateUniform(gen,b,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsR = 0;
    cudaEventElapsedTime(&millisecondsR, start, stop);
    printf("The elapsed time of random in microseconds is %f\n", millisecondsR*1000);
    cudaEventRecord(start);
    vectorAdd<<<(n + 255) / 256, 256>>>(n, a, b, d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsK = 0;
    cudaEventElapsedTime(&millisecondsK, start, stop);
    // cudaEventDestroy(startK);
    // cudaEventDestroy(stopK);
    printf("The elapsed time of kernel in microseconds is %f\n", millisecondsK*1000);
    //cuBLAS handle
    cublasHandle_t cuHandle;
    //use cuda runtime measure the time
    
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop); 

    cublasCreate_v2(&cuHandle);
    float alpha = 1.0;
    cudaEventRecord(start);
    cublasSaxpy_v2(cuHandle, n, &alpha, a, 1, b, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("The elapsed time of cublas in microseconds is %f\n", milliseconds*1000);


    //check the result  of kernel. but b and d is device memory, need kernel function do that. 
    //use kernel function to compare the result of kernel and cublas

    //measure the time of compare kernel function
    cudaEventRecord(start);
    int * areEqual;
    cudaMalloc(&areEqual, sizeof(int));
    compareVectorsKernel<<<(n + 255) / 256, 256>>>(d, b, areEqual, n);
    cudaDeviceSynchronize();
    int h_areEqual;
    cudaMemcpy(&h_areEqual, areEqual, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsC = 0;
    cudaEventElapsedTime(&millisecondsC, start, stop);
    printf("The elapsed time of compare kernel in microseconds is %f\n", millisecondsC*1000);
    if (!h_areEqual) {
        printf("The kernel and cublas results are equal\n");
    } else {
        printf("The kernel and cublas results are not equal\n");
    }
    // create two host vector to store the result of kernel and cublas
    // measure the time of compare in CPU and move the result of kernel and cublas to host memory.
    cudaEventRecord(start);
    float *h_d = (float*)malloc(n * sizeof(float));
    float *h_b = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_d, d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, b, n * sizeof(float), cudaMemcpyDeviceToHost);
    bool state=false;
    for (int i = 0; i < n; i++) {
        if (h_d[i] != h_b[i]) {
            state = true;
            break;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsH = 0;
    cudaEventElapsedTime(&millisecondsH, start, stop);
    printf("The elapsed time of compare in CPU in microseconds is %f\n", millisecondsH*1000);
    if(state)
    {
        printf("The kernel and cublas results are not equal\n");
    }else
    {
        printf("The kernel and cublas results are equal\n");
    }
    curandDestroyGenerator(gen);
    printf("program reach here %d\n",__LINE__);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);   
    cublasDestroy_v2(cuHandle);
    cudaFree(a);
    cudaFree(b);
    cudaFree(d);
    free(h_d);
    free(h_b);
    return 0;
}