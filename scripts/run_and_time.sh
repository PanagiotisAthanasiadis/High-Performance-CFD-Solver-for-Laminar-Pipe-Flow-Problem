#!/bin/bash

# This script builds the project and then runs the executable with timing.

# Exit on error
set -e

# Build the project using the existing build script
echo "--- Building the project ---"
./scripts/build_with_clang.sh

# Run the executable with timing
echo "--- Running the executable with timing ---"
time ./build/navier_stokes_gpu
