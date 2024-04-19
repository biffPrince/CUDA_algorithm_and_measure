#include <stdio.h>
// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %u\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %u\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i != 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i != 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("cache size:                    %d\n", devProp.l2CacheSize);
    printf("Total constant memory:         %u\n", devProp.totalConstMem);
    printf("Total global   memory:         %u\n", devProp.totalGlobalMem);
    printf("Texture alignment:             %u\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Size of SM:                    %d\n", devProp.streamPrioritiesSupported);
    printf("thread of SM :                 %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("thread of block:               %d\n", devProp.maxThreadsPerBlock);
    printf("warp size:                     %d\n", devProp.warpSize);
    printf("memory bus width size:         %d\n", devProp.memoryBusWidth);
    printf("memory clock rate :            %d\n", devProp.memoryClockRate);
    printf("what is this ?????:            %d\n", devProp.canMapHostMemory);
    printf("what is this ?????:            %d\n", devProp.accessPolicyMaxWindowSize);
    printf("L2 cache size is  :            %d\n", devProp.l2CacheSize);
    printf("L2 cache size is  :            %d\n", devProp.globalL1CacheSupported);
    printf("what is this ?????:            %d\n", devProp.persistingL2CacheMaxSize);
    printf("what is this ?????:            %d\n", devProp.localL1CacheSupported);
    printf("what is this ?????:            %d\n", devProp.globalL1CacheSupported);
    printf("what is this ?????:            %d\n", devProp.l2CacheSize);
    printf("concurrent kernel :            %d\n", devProp.concurrentKernels);
    printf("concurrent kernel :            %d\n", devProp.asyncEngineCount);
    printf("unified address   :            %d\n", devProp.unifiedAddressing);
    printf("unified address   :            %d\n", devProp.computeMode);
    printf("integrated compute:            %d\n", devProp.integrated);
    printf("integrated compute:            %d.%d\n", devProp.major,devProp.minor);
    printf("integrated compute:            %d\n", devProp.directManagedMemAccessFromHost);
    printf("integrated compute:            %d\n", devProp.concurrentManagedAccess);
    printf("integrated compute:            %d\n", devProp.managedMemory);
    printf("integrated compute:            %s\n", devProp.name);
    printf("what   maxGridSize:            %d\n", devProp.maxGridSize);

    // printf("what is this ?????:            %d\n", devProp.);
}

int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i != devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
    
    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);

    return 0;
}