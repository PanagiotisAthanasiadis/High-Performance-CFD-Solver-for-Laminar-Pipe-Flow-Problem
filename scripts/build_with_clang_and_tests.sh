#!/bin/bash

# This script builds the navier_stokes_project.
# It tries to find a clang compiler between version 10 and 15.(The nvidia compiler demands it.)
# If a suitable clang is found, it's used with the Ninja build system.
# Otherwise, it falls back to gcc.

# Exit on error
set -e

# --- Compiler Detection ---
CLANG_EXEC=""
CLANGPP_EXEC=""

echo "--- Searching for a suitable clang compiler (version 10-15) ---"
for v in {15..10}; do
    potential_clang="clang-$v"
    if command -v "$potential_clang" >/dev/null 2>&1; then
        echo "Found $potential_clang, checking version..."
        ver_num=$($potential_clang --version | head -n 1 | sed -n 's/.*version \([0-9]\+\).*/\1/p')
        if [ -n "$ver_num" ] && [ "$ver_num" -ge 10 ] && [ "$ver_num" -le 15 ]; then
            CLANG_EXEC="$potential_clang"
            CLANGPP_EXEC="/usr/local/llvm$v/bin/clang++"
            echo "Found suitable clang: ${CLANG_EXEC} (version ${ver_num})"
            break
        fi
    fi
done

if [ -z "$CLANG_EXEC" ]; then
    if command -v clang >/dev/null 2>&1; then
        echo "Found clang, checking version..."
        ver_num=$(clang --version | head -n 1 | sed -n 's/.*version \([0-9]\+\).*/\1/p')
        if [ -n "$ver_num" ] && [ "$ver_num" -ge 10 ] && [ "$ver_num" -le 15 ]; then
            CLANG_EXEC="clang"
            CLANGPP_EXEC="clang++"
            echo "Found suitable clang: ${CLANG_EXEC} (version ${ver_num})"
        fi
    fi
fi

# --- Build Configuration ---
PROJECT_DIR="."
BUILD_DIR="${PROJECT_DIR}/build"

CMAKE_ARGS=("-G" "Ninja" "-S" "${PROJECT_DIR}" "-B" "${BUILD_DIR}" "-DBUILD_TESTING=ON")

if [ -n "$CLANG_EXEC" ]; then
    echo "--- Configuring with clang ---"
    CMAKE_ARGS+=("-D" "CMAKE_CXX_COMPILER=${CLANGPP_EXEC}")
    CMAKE_ARGS+=("-D" "CMAKE_CUDA_HOST_COMPILER=${CLANGPP_EXEC}")
else
    echo "--- No suitable clang found, falling back to gcc ---"
    CMAKE_ARGS+=("-D" "CMAKE_CXX_COMPILER=g++")
    CMAKE_ARGS+=("-D" "CMAKE_CUDA_HOST_COMPILER=g++")
fi


# --- Build Process ---
# Clean and create build directory
echo "--- Cleaning and creating build directory ---"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Configure with CMake
echo "--- Configuring with CMake ---"
cmake "${CMAKE_ARGS[@]}"

# Build with Ninja
echo "--- Building with Ninja ---"
cmake --build "${BUILD_DIR}" #-j 4 #for parallel build if you need it 

echo "--- Build complete! The executable is in ${BUILD_DIR}/ ---"
