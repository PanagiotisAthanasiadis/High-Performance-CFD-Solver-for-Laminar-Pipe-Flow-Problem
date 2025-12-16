#pragma once

#include <vector>
#include <tuple>

void uv_velocity_single(float *out, const float Re, float *y,
                       const int xN, const int yN, const int zN,
                       const float *u_inlet,
                       const float dx, const float dy, const float dz);

std::pair<std::vector<float>, std::tuple<int*, int*, float*, int>>
Residuals_Sparse_Jacobian_Split(
    const float Re, float *y,
    const int xN, const int yN, const int zN,
    const float *u_inlet,
    const float dx, const float dy, const float dz);
