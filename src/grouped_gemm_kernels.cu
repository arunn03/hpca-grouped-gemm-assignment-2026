#include "grouped_gemm.h"

// ================================================================
// HPCA 2026: THIS IS THE ONLY FILE STUDENTS MODIFY AND SUBMIT.
// ================================================================
//
// Implement all five variants below. You may add kernels, device
// functions, constants, and helper functions IN THIS FILE.
//
// The provided driver allocates and initializes all buffers.
// V1-V3: write GEMM results to p.z; apply bias/ReLU in a separate
//        kernel to produce p.y. (One pair of launches per nonempty group.)
// V4:    fuse the epilogue; write p.y without an intermediate GEMM
//        result; launch one fused kernel per nonempty group.
// V5:    retain V4's fused computation; process all nonempty groups
//        with exactly one CUDA kernel launch.
// Bonus: optional compact scheduling, also with one launch.
//
// p.h_row_offsets is HOST memory. Device kernels must use
// p.d_row_offsets, or host-derived scalar kernel arguments.
// Never dereference p.h_row_offsets on the GPU.
//
// The TODO return code is deliberate: unimplemented variants must
// fail visibly, not silently produce incorrect or placeholder output.

namespace {

constexpr int SMEM_TILE = 16;
constexpr int REG_TILE_X = 4;
constexpr int REG_TILE_Y = 4;

__host__ __device__ int div_up(int a, int b) { return (a + b - 1) / b; }

__global__ void v1_gemm_kernel(const float* x, const float* w,
                                     float* z, int row_begin, int m,
                                     int k, int n, int group) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += x[(row_begin + row) * k + kk] *
               w[(group * k + kk) * n + col];
    }
    z[(row_begin + row) * n + col] = acc;
}

__global__ void v1_pointwise_kernel(const float* z, const float* bias,
                                          float* y, int row_begin, int m,
                                          int n, int group) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    const int col = idx % n;
    const int offset = row_begin * n + idx;
    const float val = z[offset] + bias[group * n + col];
    y[offset] = fmaxf(0.0f, val);
}

__global__ void v2_gemm_kernel(const float* x, const float* w,
                                     float* z, int row_begin, int m,
                                     int k, int n, int group) {
    __shared__ float x_s[SMEM_TILE][SMEM_TILE];
    __shared__ float w_s[SMEM_TILE][SMEM_TILE];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;
    
    for (unsigned int tile=0; tile<div_up(k, SMEM_TILE); tile++) {
        int x_ind = tile * SMEM_TILE + threadIdx.x;
        int w_ind = tile * SMEM_TILE + threadIdx.y;

        x_s[threadIdx.y][threadIdx.x] = (row < m && x_ind < k) ? x[(row_begin + row) * k + x_ind] : 0.0f;
        w_s[threadIdx.y][threadIdx.x] = (w_ind < k && col < n) ? w[(group * k + w_ind) * n + col] : 0.0f;

        __syncthreads();

        for (int kk = 0; kk < SMEM_TILE; ++kk) {
            acc += x_s[threadIdx.y][kk] * w_s[kk][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < m && col < n) z[(row_begin + row) * n + col] = acc;
}

__global__ void v2_pointwise_kernel(const float* z, const float* bias,
                                          float* y, int row_begin, int m,
                                          int n, int group) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    const int col = idx % n;
    const int offset = row_begin * n + idx;
    const float val = z[offset] + bias[group * n + col];
    y[offset] = fmaxf(0.0f, val);
}

__global__ void v3_gemm_kernel(const float* x, const float* w,
                                     float* z, int row_begin, int m,
                                     int k, int n, int group) {
    __shared__ float x_s[SMEM_TILE][SMEM_TILE+SMEM_TILE/2];
    __shared__ float w_s[SMEM_TILE][SMEM_TILE+SMEM_TILE/2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int blockRow = blockIdx.y * blockDim.y * 2;
    const int blockCol = blockIdx.x * blockDim.x * 2;
    const int row0 = blockRow + ty * 2;
    const int col0 = blockCol + tx * 2;

    float acc[2][2] = {};
    float x_r[2];
    float w_r[2];

    for (int tile = 0; tile < div_up(k, SMEM_TILE); tile++) {

        #pragma unroll
        for (int i=0; i<2; i++) {
            #pragma unroll
            for (int j=0; j<2; j++) {
                const int r = ty + SMEM_TILE/2*i;
                const int c = tx + SMEM_TILE/2*j;
                const int x_ind = tile * SMEM_TILE + c;
                const int w_ind = tile * SMEM_TILE + r;
        
                x_s[r][c] = (blockRow + r < m && x_ind < k) ? x[(row_begin + blockRow + r) * k + x_ind] : 0.0f;
                w_s[r][c] = (w_ind < k && blockCol + c < n) ? w[(group * k + w_ind) * n + blockCol + c] : 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk=0; kk<SMEM_TILE; kk++) {
            
            #pragma unroll
            for (int i=0; i<2; i++) {
                x_r[i] = x_s[ty*2 + i][kk];
                w_r[i] = w_s[kk][tx*2 + i];
            }

            #pragma unroll
            for (int i=0; i<2; i++) {
                #pragma unroll
                for (int j=0; j<2; j++) {
                    acc[i][j] += x_r[i] * w_r[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i=0; i<2; i++) {
        #pragma unroll
        for (int j=0; j<2; j++) {
            if (row0 + i < m && col0 + j < n) {
                z[(row_begin + row0 + i) * n + col0 + j] = acc[i][j];
            }
        }
    }
}

__global__ void v3_pointwise_kernel(const float* z, const float* bias,
                                          float* y, int row_begin, int m,
                                          int n, int group) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    const int col = idx % n;
    const int offset = row_begin * n + idx;
    const float val = z[offset] + bias[group * n + col];
    y[offset] = fmaxf(0.0f, val);
}

} // namespace

cudaError_t launch_v1(const GroupedGemmProblem& p, cudaStream_t stream) {
    for (int g = 0; g < p.groups; ++g) {
        const int begin = p.h_row_offsets[g];
        const int m = p.h_row_offsets[g + 1] - begin;
        if (m == 0) continue;

        // One thread calculates one GEMM output, then a separate
        // pointwise pass writes the result of bias and ReLU.
        v1_gemm_kernel<<<dim3(div_up(p.n, 128), m), 128, 0, stream>>>(
            p.x, p.w, p.z, begin, m, p.k, p.n, g);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;

        v1_pointwise_kernel<<<div_up(m * p.n, 256), 256, 0, stream>>>(
            p.z, p.bias, p.y, begin, m, p.n, g);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

cudaError_t launch_v2(const GroupedGemmProblem& p, cudaStream_t stream) {
    for (int g = 0; g < p.groups; ++g) {
        const int begin = p.h_row_offsets[g];
        const int m = p.h_row_offsets[g + 1] - begin;
        if (m == 0) continue;

        // One thread calculates one GEMM output, then a separate
        // pointwise pass writes the result of bias and ReLU.
        v2_gemm_kernel<<<dim3(div_up(p.n, SMEM_TILE), div_up(m, SMEM_TILE)), dim3(SMEM_TILE, SMEM_TILE), 0, stream>>>(
            p.x, p.w, p.z, begin, m, p.k, p.n, g);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;

        v2_pointwise_kernel<<<div_up(m * p.n, 256), 256, 0, stream>>>(
            p.z, p.bias, p.y, begin, m, p.n, g);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

cudaError_t launch_v3(const GroupedGemmProblem& p, cudaStream_t stream) {
    for (int g = 0; g < p.groups; ++g) {
        const int begin = p.h_row_offsets[g];
        const int m = p.h_row_offsets[g + 1] - begin;
        if (m == 0) continue;

        // One thread calculates one GEMM output, then a separate
        // pointwise pass writes the result of bias and ReLU.
        v3_gemm_kernel<<<dim3(div_up(p.n, SMEM_TILE), div_up(m, SMEM_TILE)), dim3(SMEM_TILE/2, SMEM_TILE/2), 0, stream>>>(
            p.x, p.w, p.z, begin, m, p.k, p.n, g);
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) return status;

        v3_pointwise_kernel<<<div_up(m * p.n, 256), 256, 0, stream>>>(
            p.z, p.bias, p.y, begin, m, p.n, g);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

cudaError_t launch_v4(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // TODO
}

cudaError_t launch_v5(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // TODO
}

cudaError_t launch_bonus(const GroupedGemmProblem& p, cudaStream_t stream) {
    (void)p;
    (void)stream;
    return cudaErrorNotSupported; // OPTIONAL
}
