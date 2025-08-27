#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#include "convolution_gpu.h"

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status_ = call;                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "CUBLAS error (%s:%d): %s, %s\n", __FILE__, __LINE__,  \
              cublasGetStatusName(status_), cublasGetStatusString(status_)); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)


__global__ void naive_kernel(float *I, float *F, float *O, int N, int C,
                                   int H, int W, int K, int R, int S, int pad_h,
                                   int pad_w, int stride_h, int stride_w,
                                   int dilation_h, int dilation_w) {
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  const int on = blockDim.x * blockIdx.x + threadIdx.x;
  const int oc = blockDim.y * blockIdx.y + threadIdx.y;

  if (on >= ON || oc >= OC) return;

  for (int oh = 0; oh < OH; ++oh) {
    for (int ow = 0; ow < OW; ++ow) {
      float sum = 0;
      for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
          for (int s = 0; s < S; ++s) {
            const int n = on;
            const int h = oh * stride_h - pad_h + r * dilation_h;
            const int w = ow * stride_w - pad_w + s * dilation_w;
            const int k = oc;
            if (h < 0 || h >= H || w < 0 || w >= W) continue;
            sum += I[((n * C + c) * H + h) * W + w] *
                   F[((k * C + c) * R + r) * S + s];
          }
        }
      }
      O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
    }
  }
}

/* Im2Col GPU Kernel */
__global__ void Im2Col_kernel(float *in, float *out, 
                              size_t N, size_t C, size_t H, size_t W, 
                              size_t R, size_t S, size_t OH, size_t OW, 
                              size_t pad_h, size_t pad_w, 
                              size_t stride_h, size_t stride_w,
                              size_t dilation_h, size_t dilation_w) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= C * R * S * N * OH * OW) return;
  int ow = tid % OW; tid /= OW;
  int oh = tid % OH; tid /= OH;
  int n = tid % N; tid /= N;
  int s = tid % S; tid /= S;
  int r = tid % R; tid /= R;
  int c = tid;

  const int h = oh * stride_h - pad_h + r * dilation_h;
  const int w = ow * stride_w - pad_w + s * dilation_w;

  if (h >= H || w >= W) {
    out[c * R * S * N * OH * OW + r * S * N * OH * OW + s * N * OH * OW + 
      n * OH * OW + oh * OW + ow] = 0;
  }
  else {
    out[c * R * S * N * OH * OW + r * S * N * OH * OW + s * N * OH * OW + 
      n * OH * OW + oh * OW + ow] = in[n * C * H * W + c * H * W + h * W + w];
  }
}

/* Col2Im GPU Kernel */
__global__ void Col2Im_kernel(float *in, float *out, size_t N, size_t K, 
                              size_t OH, size_t OW) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N * K * OH * OW) return;
  int ow = tid % OW; tid /= OW;
  int oh = tid % OH; tid /= OH;
  int k = tid % K; tid /= K;
  int n = tid;

  out[n * K * OH * OW + k * OH * OW + oh * OW + ow] = 
    in[k * N * OH * OW + n * OH * OW + oh * OW + ow];
}

static float *I_gpu, *F_gpu, *O_gpu;
static float *BUF1, *BUF2; 
static cublasHandle_t handle;

void convolution_gpu_initialize(int N, int C, int H, int W, int K, int R, int S,
                                int pad_h, int pad_w, int stride_h,
                                int stride_w, int dilation_h, int dilation_w) {
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMalloc(&I_gpu, N * C * H * W * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F_gpu, K * C * R * S * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O_gpu, ON * OC * OH * OW * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&BUF1, C * R * S * N * OH * OW * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&BUF2, K * N * OH * OW * sizeof(float)));

  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
}

void convolution_gpu(float *I, float *F, float *O, int N, int C, int H, int W,
                     int K, int R, int S, int pad_h, int pad_w, int stride_h,
                     int stride_w, int dilation_h, int dilation_w) {
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMemcpy(I_gpu, I, N * C * H * W * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F_gpu, F, K * C * R * S * sizeof(float),
                        cudaMemcpyHostToDevice));

  // /* Naive kernel */                     
  // dim3 blockDim(32, 32);
  // dim3 gridDim((N + 32 - 1) / 32, (K + 32 - 1) / 32);
  // naive_kernel<<<gridDim, blockDim>>>(I_gpu, F_gpu, O_gpu, N, C, H, W, K,
  //                                           R, S, pad_h, pad_w, stride_h,
  //                                           stride_w, dilation_h, dilation_w);

  /* 
   * Im2Col: 
   * in [N, C, H, W] 
   * -> BUF1 [C, R, S, N, OH, OW] 
   */
  int numThreads = 1024;
  int numBlocks = (C * R * S * N * OH * OW + numThreads - 1) / numThreads;
  dim3 blockDim(numThreads);
  dim3 gridDim(numBlocks);
  Im2Col_kernel<<<gridDim, blockDim>>>(I_gpu, BUF1, N, C, H, W, R, S, OH, OW,
                                       pad_h, pad_w, stride_h, stride_w,
                                       dilation_h, dilation_w);

  /* 
   * cuBLAS GEMM
   * weight [K, C*R*S] x BUF1 [C*R*S, N*OH*OW] 
   * -> BUF2 [K, N*OH*OW] 
   */
  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N * OH * OW, K, C * R * S, 
                           &alpha, BUF1, N * OH * OW, F_gpu, C * R * S, 
                           &beta, BUF2, N * OH * OW));

  /* 
   * Col2Im: 
   * BUF2 [K, N, OH, OW] -> out [N, K, OH, OW] 
   */
  numThreads = 1024;
  numBlocks = (N * K * OH * OW + numThreads - 1) / numThreads;
  blockDim = dim3(numThreads);
  gridDim = dim3(numBlocks);
  Col2Im_kernel<<<gridDim, blockDim>>>(BUF2, O_gpu, N, K, OH, OW);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, O_gpu, ON * OC * OH * OW * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void convolution_gpu_finalize(int N, int C, int H, int W, int K, int R, int S,
                              int pad_h, int pad_w, int stride_h, int stride_w,
                              int dilation_h, int dilation_w) {
  CHECK_CUDA(cudaFree(I_gpu));
  CHECK_CUDA(cudaFree(F_gpu));
  CHECK_CUDA(cudaFree(O_gpu));
}