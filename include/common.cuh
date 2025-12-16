#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

// ============================================================================ 
// ERROR CHECKING AND UTILITIES
// ============================================================================ 

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error " << err << " at "           \
                  << __FILE__ << ":" << __LINE__ << " -> "    \
                  << cudaGetErrorString(err) << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while(0)

// Memory alignment utility
inline size_t align_size(size_t size, size_t alignment = 256) {
    return ((size + alignment - 1) / alignment) * alignment;
}

// ============================================================================ 
// CONSISTENT INDEX FUNCTIONS
// ============================================================================ 

// Standard 3D indexing: (i, j, k) -> linear index
__device__ __host__ inline int idx_3d(int i, int j, int k, int sizeY, int sizeZ) {
    return (i * sizeY + j) * sizeZ + k;
}

// Batched 3D indexing with grain dimension
__device__ __host__ inline int idx_3d_batch(int i, int j, int k, int l,
                                             int sizeY, int sizeZ, int grain) {
    return ((i * sizeY + j) * sizeZ + k) * grain + l;
}
