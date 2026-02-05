#!/bin/bash
# This script builds the navier_stokes_project.
# It tries to find a clang++ compiler between version 10 and 14 in the PATH.
# (The nvidia compiler demands it.)
# If a suitable clang++ is found, it's used with the Ninja build system.
# Otherwise, it falls back to gcc.

# Exit on error
set -e

# --- Compiler Detection ---
CLANG_EXEC=""
CLANGPP_EXEC=""

echo "--- Searching for a suitable clang++ compiler (version 10-14) in PATH ---"

for v in {14..10}; do
    potential_clangpp="clang++-$v"
    if command -v "$potential_clangpp" >/dev/null 2>&1; then
        echo "Found $potential_clangpp, checking version..."
        ver_num=$($potential_clangpp --version | head -n 1 | sed -n 's/.*version \([0-9]\+\).*/\1/p')
        if [ -n "$ver_num" ] && [ "$ver_num" -ge 10 ] && [ "$ver_num" -le 14 ]; then
            CLANGPP_EXEC="$potential_clangpp"
            CLANG_EXEC="clang-$v"
            echo "Found suitable clang++: ${CLANGPP_EXEC} (version ${ver_num})"
            break
        fi
    fi
done

if [ -z "$CLANGPP_EXEC" ]; then
    if command -v clang++ >/dev/null 2>&1; then
        echo "Found clang++, checking version..."
        ver_num=$(clang++ --version | head -n 1 | sed -n 's/.*version \([0-9]\+\).*/\1/p')
        if [ -n "$ver_num" ] && [ "$ver_num" -ge 10 ] && [ "$ver_num" -le 14 ]; then
            CLANGPP_EXEC="clang++"
            CLANG_EXEC="clang"
            echo "Found suitable clang++: ${CLANGPP_EXEC} (version ${ver_num})"
        fi
    fi
fi

# --- Build Configuration ---
PROJECT_DIR="."
BUILD_DIR="${PROJECT_DIR}/build"

CMAKE_ARGS=("-G" "Ninja"
            "-S" "${PROJECT_DIR}"
            "-B" "${BUILD_DIR}"
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=1"
            "-DCMAKE_BUILD_TYPE=Debug"
            "-DBUILD_TESTING=ON")

if [ -n "$CLANGPP_EXEC" ]; then
    echo "--- Configuring with clang++ ---"
    CMAKE_ARGS+=("-D" "CMAKE_CXX_COMPILER=${CLANGPP_EXEC}")
    CMAKE_ARGS+=("-D" "CMAKE_CUDA_HOST_COMPILER=${CLANGPP_EXEC}")
    
    # Find libomp.so in system paths
    LIBOMP_PATH=""
    for path in /usr/lib/llvm-*/lib/libomp.so /usr/lib/x86_64-linux-gnu/libomp.so /usr/lib/libomp.so /usr/local/lib/libomp.so; do
        if [ -f "$path" ]; then
            LIBOMP_PATH="$path"
            echo "Found libomp at: $LIBOMP_PATH"
            break
        fi
    done
    
    if [ -n "$LIBOMP_PATH" ]; then
        # Set OpenMP flags explicitly for clang
        CMAKE_ARGS+=("-D" "OpenMP_CXX_FLAGS=-fopenmp")
        CMAKE_ARGS+=("-D" "OpenMP_CXX_LIB_NAMES=omp")
        CMAKE_ARGS+=("-D" "OpenMP_omp_LIBRARY=${LIBOMP_PATH}")
    else
        echo "WARNING: libomp.so not found. OpenMP may not work correctly."
        CMAKE_ARGS+=("-D" "OpenMP_CXX_FLAGS=-fopenmp")
    fi
else
    echo "--- No suitable clang++ found, falling back to gcc ---"
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