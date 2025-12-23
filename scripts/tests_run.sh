#!/bin/bash

# This script builds the project and then runs the executable with timing.

# Exit on error
set -e

# Build the project using the existing build script
echo "--- Building the project with tests---"
./scripts/build_with_clang_and_tests.sh

# Run the executable with timing
echo "--- Running the tests ---"
cd ./build/
ctest