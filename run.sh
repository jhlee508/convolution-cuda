#!/bin/bash

# srun -N 1 --partition PV-Short --exclusive \
#   ./main $@

#########################################################
Device=0 # (0 for CPU, 1 for GPU, 2 for GPU with cuDNN)
N=2
C=128
H=160
W=160
K=256
R=3
S=3
#########################################################

srun -N 1 --partition PV --exclusive --gres=gpu:1 \
  ./main $@ -v -n 5 \
  $Device \
  $N $C $H $W \
  $K $R $S \
  1 1 \
  1 1 \
  1 1 
