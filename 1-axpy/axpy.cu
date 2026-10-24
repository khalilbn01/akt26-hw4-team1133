
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// GPU AXPY kernel
__global__ void axpy_kernel(int n, float alpha,
                            const float* x, float* y)
{
    // calculate global thread ID
    int gtx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtx < n) {
        y[ gtx ] = alpha * x[ gtx ] + y[ gtx ];
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation
void axpy_cpu(int n, float alpha,
              const float* x, float* y)
{
    for (int i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Maximum elementwise relative error
double max_relative_error(int n,
                          const float* result,
                          const float* reference)
{
    double max_rel = 0.0;

    for (int i = 0; i < n; ++i) {
        double diff = std::fabs(
            (double)result[i] - (double)reference[i]);
        double denom = std::fabs((double)reference[i]);

        double rel;
        if (denom > 0.0) {
            rel = diff / denom;
        } else {
            rel = (diff == 0.0) ? 0.0 : INFINITY;
        }

        max_rel = std::max(max_rel, rel);
    }

    return max_rel;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    const int n       = (argc > 1) ? std::atoi(argv[1]) : 10000;
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 10;
    const float alpha = 2.0f;

    // --------------------------------------------------------
    // GPU AXPY: kernel configuration
    // --------------------------------------------------------
    const int threads_x = 1;
    const int grid_x    = (n + threads_x - 1) / threads_x;

    if (n <= 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: %s [N > 0] [repeats > 0]\n",
                     argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize input vectors on the CPU
    std::vector<float> h_x(n), h_y0(n);
    std::vector<float> h_ref(n);
    std::vector<float> h_gpu(n), h_cublas(n);

    for (int i = 0; i < n; ++i) {
        h_x[i]  = std::sin(0.001f * i);
        h_y0[i] = std::cos(0.002f * i);
    }

    // compute using CPU in h_ref
    h_ref = h_y0;   // copy h_y0 into h_ref
    axpy_cpu(n, alpha, h_x.data(), h_ref.data());

    // Allocate GPU memory
    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, (size_t)n * sizeof(float));
    cudaMalloc(&d_y, (size_t)n * sizeof(float));

    // copy h_x into d_x
    cudaMemcpy(d_x, h_x.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    // GPU information
    std::printf("N = %d, alpha = %.1f, repeats = %d\n", n, alpha, repeats);
    std::printf("Launch configuration: grid=(%d,1,1), block=(%d,1,1)\n\n", grid_x, threads_x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(grid_x, 1, 1);
    dim3 threads(threads_x, 1, 1);

    // --------------------------------------------------------
    // GPU AXPY kernel: warm-up and timing
    // --------------------------------------------------------
    cudaMemcpy(d_y, h_y0.data(), (size_t)n * sizeof(float),
               cudaMemcpyHostToDevice);

    // warmup run
    axpy_kernel<<<grid, threads>>>(n, alpha, d_x, d_y);
    cudaDeviceSynchronize();

    // start timing
    cudaEventRecord(start);
    for (int r = 0; r < repeats; ++r) {
        axpy_kernel<<<grid, threads>>>(n, alpha, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float custom_ms = 0.0f;
    cudaEventElapsedTime(&custom_ms, start, stop);
    double custom_us = custom_ms * 1000.0 / repeats;

    // Correctness run: restore initial y and execute once
    cudaMemcpy(d_y, h_y0.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    axpy_kernel<<<grid, threads>>>(n, alpha, d_x, d_y);
    cudaMemcpy(h_gpu.data(), d_y, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // cuBLAS SAXPY: warm-up and timing
    // --------------------------------------------------------
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemcpy(d_y, h_y0.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    // warmup run
    cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);
    cudaDeviceSynchronize();

    // start timing
    cudaEventRecord(start);
    for (int r = 0; r < repeats; ++r) {
        cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    double cublas_us = cublas_ms * 1000.0 / repeats;

    // Correctness run: restore initial y and execute once
    cudaMemcpy(d_y, h_y0.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);
    cudaMemcpy(h_cublas.data(), d_y, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    // --------------------------------------------------------
    // Performance and maximum relative error
    // AXPY counts 2 floating-point operations per element.
    // --------------------------------------------------------
    const double flops = 2.0 * n;

    double custom_gflops = flops / (custom_us * 1.0e3);
    double cublas_gflops = flops / (cublas_us * 1.0e3);

    double custom_error = max_relative_error( n, h_gpu.data(), h_ref.data() );
    double cublas_error = max_relative_error( n, h_cublas.data(), h_ref.data() );

    std::printf("%-20s %12s %12s %20s\n",
                "Implementation", "Time (us)", "GFLOP/s",
                "Max relative error");
    std::printf("--------------------------------------------------------------------\n");
    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "GPU kernel", custom_us, custom_gflops,
                custom_error);
    std::printf("%-20s %12.3f %12.3f %20.6e\n",
                "cuBLAS SAXPY", cublas_us, cublas_gflops,
                cublas_error);

    std::printf("\nSpeedup against cuBLAS: %.3fx\n",
                cublas_us / custom_us);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);

    return EXIT_SUCCESS;
}
