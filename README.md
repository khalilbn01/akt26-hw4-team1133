# AKT26 HW4: CUDA kernels (Team 1133)

Tested on Colab, Tesla T4 (sm_75). Build: `make sm=75`.

## 3-ger: rank-1 update A = A + alpha * x * y^T
Each thread updates 4 consecutive rows (one `float4`) of 4 consecutive columns; a block of 32x4 threads
covers 128 rows x 16 columns, so a warp reads 512 contiguous bytes (coalesced). A scalar path handles
any `m` that is not a multiple of 4.

| Size (m x n) | Custom (us) | cuBLAS (us) | Speedup | Error norm |
|---|---|---|---|---|
| 4000 x 2000 | 288.231 | 310.300 | 1.077x | 5.86e-08 |
| 4001 x 1999 | 323.709 | 355.165 | 1.097x | 5.86e-08 |
| 7 x 3 | 3.033 | 2.823 | 0.931x | 5.96e-08 |
| 130 x 33 | 4.219 | 3.235 | 0.767x | 6.11e-08 |

On tiny matrices the run time is dominated by kernel-launch overhead, so cuBLAS is on par.

## 4-gemm: batched GEMM, C_i = A_i * B_i, n <= 32
One thread-block per matrix with n x n threads; A_i and B_i are staged in shared memory
(`sA` stored transposed and `sB` padded by one column to avoid bank conflicts).

| n | Batch | Custom (us) | cuBLAS (us) | Speedup | Max fwd error |
|---|---|---|---|---|---|
| 16 | 1000 | 22.401 | 43.561 | 1.945x | 2.44e-08 |
| 7 | 5000 | 39.055 | 114.585 | 2.934x | 3.14e-08 |
| 1 | 100 | 2.996 | 7.014 | 2.342x | 0 |
| 17 | 777 | 20.343 | 75.111 | 3.692x | 2.08e-08 |
| 8 | 100000 | 479.561 | 1845.105 | 3.847x | 3.41e-08 |
| 32 | 1000 | 159.035 | 123.695 | 0.778x | 1.34e-08 |

All runs print `TEST PASS` (tolerance 1e-6). For n = 32 (1024 threads per block) cuBLAS is faster;
a possible improvement is computing several output elements per thread.
