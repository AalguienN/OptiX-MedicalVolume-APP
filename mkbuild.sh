#!/bin/bash

PROJECT_DIR="$HOME/projects/TFM/optix_clone"
BUILD_DIR="$PROJECT_DIR/build"

cd "$PROJECT_DIR" || {
    echo "Project not found"
    exit 1
}

rm -rf build
mkdir build
cd build || exit 1

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCUDA_ARCH="sm_120"

make -j$(nproc)
