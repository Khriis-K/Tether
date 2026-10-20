#!/bin/sh
# Configure, build and test the C++ tracker. Usage: cpp/build.sh [Debug|Release]
set -e
cd "$(dirname "$0")"
CONFIG="${1:-Release}"
cmake -S . -B build
cmake --build build --config "$CONFIG" --parallel
ctest --test-dir build -C "$CONFIG" --output-on-failure
