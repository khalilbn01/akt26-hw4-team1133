
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef TX
#define TX 256
#endif

static_assert(TX > 0 && (TX & (TX - 1)) == 0, "TX must be a power of two");
static_assert(TX <= 1024, "TX must not exceed 1024");

////////////////////////////////////////////////////////////////////////////////
// dot-product kernel
// TX threads per block; shared-memory reduction;
// atomicAdd accumulates each block's result.
__global__ void dot_kernel(int n,
                           const float* x,
                           const float* y,
                           float* result)
{
    __shared__ float sdata[TX];

    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;

    // element-wise products
    float sum = 0.0f;
    if (gtx < n) {
        sum = x[gtx] * y[gtx];
    }

    // store element-wise product in shared memory
    sdata[tx] = sum;
    __syncthreads();

    // block-level reduction in shared memory
    // TX is a power of two.
    for (int stride = TX / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            sdata[tx] += sdata[tx + stride];
        }
        __syncthreads();
    }

    // one thread writes the result of the block using atomics
    if (tx == 0) {
        atomicAdd(result, sdata[0]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference
float dot_cpu(int n, const float* x, const float* y)
{
    float sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }

    return sum;
}

////////////////////////////////////////////////////////////////////////////////
// Relative error for the scalar dot product
double relative_error(float result, float reference)
{
    const double diff  = std::fabs((double)result - (double)reference);
    const double denom = std::fabs((double)reference);

    if (denom > 0.0) {
        return diff / denom;
    }

    return diff;
}

////////////////////////////////////////////////////////////////////////////////
// main
int main(int argc, char** argv)
{
    const int n       = (argc > 1) ? std::atoi(argv[1]) : 100000;
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 100;

    if (n <= 0 || repeats <= 0) {
        std::fprintf(stderr,
                     "Usage: %s [N > 0] [repeats > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int blocks = (n + TX - 1) / TX;

    // Initialize vectors on the CPU
    std::vector<float> h_x(n), h_y(n);

    for (int i = 0; i < n; ++i) {
        h_x[i] = std::sin(0.001f * i);
        h_y[i] = std::cos(0.002f * i);
    }

    const float h_ref = dot_cpu(n, h_x.data(), h_y.data());

    // Allocate GPU memory
    float *d_x = nullptr, *d_y = nullptr, *d_result = nullptr;

    cudaMalloc(&d_x,      (size_t)n       * sizeof(float));
    cudaMalloc(&d_y,      (size_t)n       * sizeof(float));
    cudaMalloc(&d_result, (size_t)repeats * sizeof(float));

    // send data to the GPU
    cudaMemcpy(d_x, h_x.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    std::printf("N = %d, TX = %d, blocks = %d, repeats = %d\n\n", n, TX, blocks, repeats);

    // for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --------------------------------------------------------
    // Custom kernel: warm-up and timing
    // --------------------------------------------------------

    // initialize to zero so that atomicAdd can be used
    cudaMemset(d_result, 0, (size_t)repeats * sizeof(float));

    // warmup
    dot_kernel<<<blocks, TX>>>(n, d_x, d_y, d_result);
    cudaDeviceSynchronize();

    // re-initialize to zero so that atomicAdd can be used
    cudaMemset(d_result, 0, (size_t)repeats * sizeof(float));

    // start timing
    cudaEventRecord(start);
    for (int r = 0; r < repeats; ++r) {
        dot_kernel<<<blocks, TX>>>(n, d_x, d_y, d_result + r);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float custom_ms = 0.0f;
    cudaEventElapsedTime(&custom_ms, start, stop);
    const double custom_us = custom_ms * 1000.0 / repeats;

    float h_custom = 0.0f;
    cudaMemcpy(&h_custom, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // cuBLAS dot product: cublasSdot
    // Use device pointer mode to write results to GPU memory.
    // --------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    cudaMemset(d_result, 0, (size_t)repeats * sizeof(float));

    cublasSdot(handle, n, d_x, 1, d_y, 1, d_result);
    cudaDeviceSynchronize();

    cudaMemset(d_result, 0, (size_t)repeats * sizeof(float));

    cudaEventRecord(start);
    for (int r = 0; r < repeats; ++r) {
        cublasSdot(handle, n, d_x, 1, d_y, 1, d_result + r);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    const double cublas_us = cublas_ms * 1000.0 / repeats;

    float h_cublas = 0.0f;
    cudaMemcpy(&h_cublas, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // Performance and relative error
    // Approximately 2 FLOPs per element: multiply and add.
    // --------------------------------------------------------
    const double flops = 2.0 * n;

    const double custom_gflops = flops / (custom_us * 1.0e3);
    const double cublas_gflops = flops / (cublas_us * 1.0e3);

    const double custom_error = relative_error(h_custom, h_ref);
    const double cublas_error = relative_error(h_cublas, h_ref);

    std::printf("CPU reference: %.9g\n\n", h_ref);

    std::printf("%-20s %12s %12s %20s\n",
                "Implementation", "Time (us)", "GFLOP/s",
                "Relative error");
    std::printf("--------------------------------------------------------------------\n");

    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "GPU    kernel", custom_us, custom_gflops,
                custom_error);
    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "cuBLAS Sdot", cublas_us, cublas_gflops,
                cublas_error);

    std::printf("\nSpeedup (cuBLAS time / custom time): %.3fx\n",
                cublas_us / custom_us);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);

    return 0;
}
