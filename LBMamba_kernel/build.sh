#!/bin/bash

if [ -d "build" ]; then
    rm -r build
fi

mkdir build

cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="/home/jzhang/Dev/anaconda3_2023/envs/vim"  -DOUTPUT_DIRECTORY=../../lbm/pscan_cuda/ -DCUDA_ARCHS="70;75;80" -B build
#cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="/opt/conda" -DCUDA_ARCHS="70;75;80" -DOUTPUT_DIRECTORY=../../lbm/pscan_cuda/ -B build

cmake --build build -- -j32