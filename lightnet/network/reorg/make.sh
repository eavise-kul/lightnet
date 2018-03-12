#!/usr/bin/env bash

echo "Compiling reorg layer kernels by nvcc..."
nvcc -c -o src/reorg_cuda_kernel.cu.o src/reorg_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
python3.6 build.py
