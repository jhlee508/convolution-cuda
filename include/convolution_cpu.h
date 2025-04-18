#pragma once

#include <cstddef>


void convolution_cpu(float *I, float *F, float *O, int N, int C, int H, int W,
                     int K, int R, int S, int pad_h, int pad_w, int stride_h,
                     int stride_w, int dilation_h, int dilation_w);

void convolution_cpu_initialize(int N, int C, int H, int W, int K, int R, int S,
                                int pad_h, int pad_w, int stride_h,
                                int stride_w, int dilation_h, int dilation_w);

void convolution_cpu_finalize(int N, int C, int H, int W, int K, int R, int S,
                              int pad_h, int pad_w, int stride_h, int stride_w,
                              int dilation_h, int dilation_w);