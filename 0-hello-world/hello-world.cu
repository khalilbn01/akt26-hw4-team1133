
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void hello_kernel()
{
    printf("Hello from block(%d, %d, %d) thread (%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

// Print basic GPU information
void print_gpu_info()
{
    int device = 0;
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return;
    }

    int clk_khz, memclk_khz;
    #if CUDART_VERSION >= 12000
    cudaDeviceGetAttribute( &clk_khz,    cudaDevAttrClockRate, device );
    cudaDeviceGetAttribute( &memclk_khz, cudaDevAttrMemoryClockRate, device );
    #else
    clk_khz    = prop.clockRate;
    memclk_khz = prop.memoryClockRate;
    #endif

    printf("GPU Information\n");
    printf("------------------------------\n");
    printf("Device name              : %s\n",       prop.name);
    printf("Compute capability       : %d.%d\n",    prop.major, prop.minor);
    printf("Number of SMs            : %d\n",       prop.multiProcessorCount);
    printf("GPU clock rate           : %.2f MHz\n", clk_khz / 1000.0);
    printf("Memory clock rate        : %.2f MHz\n", memclk_khz / 1000.0);
    printf("L2 cache size            : %.2f MB\n",  prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Shared memory per block  : %.2f KB\n",  prop.sharedMemPerBlock / 1024.0);
    printf("Shared memory per SM     : %.2f KB\n",  prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Global memory            : %.2f GB\n",  prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Warp size                : %d\n",       prop.warpSize);
    printf("Max threads per block    : %d\n\n",     prop.maxThreadsPerBlock);
}

int main()
{
    print_gpu_info();

    printf("Kernel Launch\n");
    printf("------------------------------\n");
    // specify #threads and #blocks
    int threads_x = 1;
    int blocks_x  = 1;

    // kernel config
    dim3 threads(threads_x, 1, 1);
    dim3 grid(blocks_x, 1, 1);

    // launch
    hello_kernel<<<grid, threads>>>();

    // Wait for kernel completion and check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
