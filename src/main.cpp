#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convolution_cpu.h"
#include "convolution_cudnn.h"
#include "convolution_gpu.h"
#include "util.h"


static bool print = false;
static bool validation = false;
static size_t T = 0;
static int N = 1;
static int C = 3;
static int H = 3;
static int W = 3;
static int K = 3;
static int R = 3;
static int S = 3;
static int pad_h = 0;
static int pad_w = 0;
static int stride_h = 1;
static int stride_w = 1;
static int dilation_h = 1;
static int dilation_w = 1;

static size_t num_iterations = 1;

static const char *convolution_type_string[] = {"CPU (sequential)", "GPU",
                                                "cuDNN"};

static void print_help(const char *prog_name) {
  printf(
      "Usage: %s [-pvh] [-n num_iterations] T N C H W K R S pad_h pad_w "
      "stride_h stride_w dilation_h dilation_w\n",
      prog_name);
  printf("Options:\n");
  printf("     -p: print tensor. (default: off)\n");
  printf("     -v: validate convolution. (default: off)\n");
  printf("     -h: print this page.\n");
  printf("     -n: number of iterations (default: 1)\n");
  printf("      T: type of convolution (default: 0)\n");
  printf("            0: CPU (sequential)\n");
  printf("            1: GPU\n");
  printf("            2: cuDNN\n");
  printf("      N: batch size (default: 1)\n");
  printf("      C: input channel size (default: 3)\n");
  printf("      H: input height (default: 3)\n");
  printf("      W: input width (default: 3)\n");
  printf("      K: output channel size (default: 3)\n");
  printf("      R: filter height (default: 3)\n");
  printf("      S: filter width (default: 3)\n");
  printf("      pad_h: top and bottom padding (default: 0)\n");
  printf("      pad_w: left and right padding (default: 0)\n");
  printf("      stride_h: vertical stride (default: 1)\n");
  printf("      stride_w: horizontal stride (default: 1)\n");
  printf("      dilation_h: vertical dilation (default: 1)\n");
  printf("      dilation_w: horizontal dilation (default: 1)\n");
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:m:")) != -1) {
    switch (c) {
      case 'p': print = true; break;
      case 'v': validation = true; break;
      case 'n': num_iterations = atoi(optarg); break;
      case 'h':
      default: print_help(argv[0]); exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: T = (size_t) atoi(argv[i]); break;
      case 1: N = (size_t) atoi(argv[i]); break;
      case 2: C = (size_t) atoi(argv[i]); break;
      case 3: H = (size_t) atoi(argv[i]); break;
      case 4: W = (size_t) atoi(argv[i]); break;
      case 5: K = (size_t) atoi(argv[i]); break;
      case 6: R = (size_t) atoi(argv[i]); break;
      case 7: S = (size_t) atoi(argv[i]); break;
      case 8: pad_h = (size_t) atoi(argv[i]); break;
      case 9: pad_w = (size_t) atoi(argv[i]); break;
      case 10: stride_h = (size_t) atoi(argv[i]); break;
      case 11: stride_w = (size_t) atoi(argv[i]); break;
      case 12: dilation_h = (size_t) atoi(argv[i]); break;
      case 13: dilation_w = (size_t) atoi(argv[i]); break;
      default: break;
    }
  }

  printf("================== Convolution Benchmark ==================\n");
  printf("- Number of iterations: %lu\n", num_iterations);
  printf("- Print tensor: %s\n", print ? "on" : "off");
  printf("- Validation: %s\n", validation ? "on" : "off");
  printf("- Convolution Type: %s\n", convolution_type_string[T]);
  printf("- Problem size: \n"
      "  -> Input (N, C, H, W) = (%d, %d, %d, %d)\n"
      "  -> Filter (K, R, S) = (%d, %d, %d)\n",
      N, C, H, W, K, R, S);
  printf(
      "  -> Padding (pad_h, pad_w) = (%d, %d)\n"
      "  -> Stride (stride_h, stride_w) = (%d, %d)\n"
      "  -> Dilation (dilation_h, dilation_w) = (%d, %d)\n",
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
  unsigned long long int FLOPS = 2ULL * N * K * H * W * C * R * S;
  unsigned long long int BYTES = (1ULL * (N * C * H * W) + 
                                  1ULL * (K * C * R * S) + 
                                  1ULL * (N * K * H * W)) * sizeof(float);
  unsigned long long int ELEMS = 1ULL * (N * C * H * W) + 
                                  1ULL * (K * C * R * S) + 
                                  1ULL * (N * K * H * W);
  printf("- Number of FLOPs: %llu\n", FLOPS);
  printf("- Number of BYTEs: %llu\n", BYTES);
  printf("- Number of ELEMs: %llu\n", ELEMS);
  printf("- FLOPs/BYTE: %.2f\n", (double)FLOPS / BYTES);
  printf("- FLOPs/ELEM: %.2f\n", (double)FLOPS / ELEMS);
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);
  fflush(stdout);

  /* Allocate and initialize tensor on CPU */
  float *I, *F, *O;
  alloc_tensor(&I, N, C, H, W);
  alloc_tensor(&F, K, C, R, S);

  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  alloc_tensor(&O, ON, OC, OH, OW);

  rand_tensor(I, N, C, H, W);
  rand_tensor(F, K, C, R, S);

  /* Initialize Convolution */
  switch (T) {
    case 0:
      convolution_cpu_initialize(N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                                 stride_w, dilation_h, dilation_w);
      break;
    case 1:
      convolution_gpu_initialize(N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                                 stride_w, dilation_h, dilation_w);
      break;
    case 2:
      convolution_cudnn_initialize(N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                                   stride_w, dilation_h, dilation_w);
      break;
  }

  /* Run few warmup iterations... */
  for (size_t i = 0; i < 3; i++) {
    zero_tensor(O, ON, OC, OH, OW);
    switch (T) {
      case 0:
        convolution_cpu(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                        stride_w, dilation_h, dilation_w);
        break;
      case 1:
        convolution_gpu(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                        stride_w, dilation_h, dilation_w);
        break;
      case 2:
        convolution_cudnn(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                          stride_w, dilation_h, dilation_w);
        break;
    }
  }

  /* Run convolution for num_iterations */
  printf("\n--------------------- Run Benchmark -----------------------\n");

  double elapsed_time_sum = 0;
  for (size_t i = 0; i < num_iterations; ++i) {
    printf("[iter %lu] ", i);
    fflush(stdout);

    zero_tensor(O, ON, OC, OH, OW);
    double elapsed_time_iter = -get_current_time();
    switch (T) {
      case 0:
        convolution_cpu(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                        stride_w, dilation_h, dilation_w);
        break;
      case 1:
        convolution_gpu(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                        stride_w, dilation_h, dilation_w);
        break;
      case 2:
        convolution_cudnn(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                          stride_w, dilation_h, dilation_w);
        break;
    }
    elapsed_time_iter += get_current_time();

    printf("%.4f s\n", elapsed_time_iter);
    elapsed_time_sum += elapsed_time_iter;
  }

  if (print) {
    printf("\n---------------------- Print Tensor -----------------------\n");
    printf("INPUT:\n");
    print_tensor(I, N, C, H, W);
    printf("FILTER:\n");
    print_tensor(F, K, C, R, S);
    printf("OUTPUT:\n");
    print_tensor(O, ON, OC, OH, OW);
  }

  if (validation) {
    printf("\n----------------------- Validation ------------------------\n");
    check_convolution(I, F, O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                      stride_w, dilation_h, dilation_w);
  }

  /* Print performance results */
  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("\n-------------------- Benchmark Summary --------------------\n");
  printf("Avg. time        : %.4f s\n", elapsed_time_avg);
  printf("Avg. performance : %.1f GFLOPS\n",
         2.0 * ON * OC * OH * OW * C * R * S / elapsed_time_avg / 1e9);

  /* Finalize convolution */
  switch (T) {
    case 0:
      convolution_cpu_finalize(N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                               stride_w, dilation_h, dilation_w);
      break;
    case 1:
      convolution_gpu_finalize(N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                               stride_w, dilation_h, dilation_w);
      break;
    case 2:
      convolution_cudnn_finalize(N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                                 stride_w, dilation_h, dilation_w);
      break;
  }

  printf("\n===========================================================\n");
  return 0;
}