#!/bin/bash

# srun -N 1 --partition PV-Short --exclusive \
#   ./main $@

stride_h=1
stride_w=1
pad_h=1
pad_w=1
dilation_h=1
dilation_w=1

#########################################################
Device=2 # (0 for CPU, 1 for GPU, 2 for cuDNN)
N=1
C=1
H=$((16384))
W=$((16384))
K=1
R=61
S=61
#########################################################

srun -N 1 --partition PV --exclusive --gres=gpu:1 \
  ./main $@ -n 5 \
  $Device \
  $N $C $H $W \
  $K $R $S \
  $stride_h $stride_w \
  $pad_h $pad_w \
  $dilation_h $dilation_w \
  # -v
