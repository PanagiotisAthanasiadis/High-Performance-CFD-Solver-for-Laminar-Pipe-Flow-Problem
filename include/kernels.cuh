#pragma once

#include <cuda_runtime.h>

// KERNEL: BUILD SOLUTION BATCH
__global__ void build_ysol_batch_kernel(
    const float* __restrict__ y,
    const float* __restrict__ h,
    float* __restrict__ ysol_batch,
    const int nCell,
    const int grain,
    const int* __restrict__ t_batch);

// KERNEL: BOUNDARY CONDITIONS INITIALIZATION (BATCHED)
__global__ void
boundary_conditions_initialization(
    const float* __restrict__ ysol_local_batch,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
    const int xN, const int yN, const int zN,
    const int grain);

// KERNEL: BOUNDARY CONDITIONS APPLY (BATCHED)
__global__ void boundary_conditions_apply(
    const float* __restrict__ u_inlet,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
    const int xN, const int yN, const int zN,
    const int grain);


// KERNEL: BOUNDARY CONDITIONS INITIALIZATION (SINGLE)
__global__ void boundary_conditions_initialization_single(
    const float* __restrict__ ysol,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
    const int xN, const int yN, const int zN);

// KERNEL: BOUNDARY CONDITIONS APPLY (SINGLE)
__global__ void boundary_conditions_apply_single(
    const float* __restrict__ u_inlet,
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    float* __restrict__ p,
    const int xN, const int yN, const int zN);

// KERNEL: UV VELOCITY (SINGLE)
__global__ void kernel_uv_velocity_single(
    float* __restrict__ out, float Re,
    float* __restrict__ u, float* __restrict__ v,
    float* __restrict__ p, float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz);

// HOST FUNCTION: RESIDUALS AND SPARSE JACOBIAN
__global__ void build_jacobian_entries(
    const float* __restrict__ out,
    const float* __restrict__ fold,
    const float* __restrict__ h,
    int* __restrict__ row_idx,
    int* __restrict__ col_idx,
    float* __restrict__ values,
    int* __restrict__ counter,
    const int nCell, const int grain, const int start);

// Momentum Kernels
__global__ void
kernel_u_momentum(
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ p, const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
    const float Re, const int sizeY, const int sizeZ);

__global__ void
kernel_v_momentum(
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ p, const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
    const float Re, const int sizeY, const int sizeZ);

__global__ void
kernel_w_momentum(
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ p, const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
    const float Re, const int sizeY, const int sizeZ);

__global__ void
kernel_continuity(
    int grain, float* __restrict__ out,
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ w,
    const int xN, const int yN, const int zN,
    const float dx, const float dy, const float dz,
    const int sizeY, const int sizeZ);
