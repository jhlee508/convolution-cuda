#include <cstdlib>

#include "convolution_cpu.h"


void convolution_cpu_initialize(int N, int C, int H, int W, int K, int R, int S,
                                int pad_h, int pad_w, int stride_h,
                                int stride_w, int dilation_h, int dilation_w) {
  // Nothing to do
  return;
}

void convolution_cpu(float *I, float *F, float *O, int N, int C, int H, int W,
                     int K, int R, int S, int pad_h, int pad_w, int stride_h,
                     int stride_w, int dilation_h, int dilation_w) {
  // Naive CPU multiplication
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
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
  }
}

void convolution_cpu_finalize(int N, int C, int H, int W, int K, int R, int S,
                              int pad_h, int pad_w, int stride_h, int stride_w,
                              int dilation_h, int dilation_w) {
  // Nothing to do
  return;
}