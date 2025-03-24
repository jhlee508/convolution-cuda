#!/bin/bash

# srun -N 1 --partition PV-Short --exclusive \
#   ./main $@

# CPU version 
# I(1, 32, 40, 40) * W(16, 3, 3) -> O(1, 32, 40, 40)
  ./main $@ -v -n 5 \
  0 \
  1 32 40 40 \
  16 3 3 \
  1 1 \
  1 1 \
  1 1 
