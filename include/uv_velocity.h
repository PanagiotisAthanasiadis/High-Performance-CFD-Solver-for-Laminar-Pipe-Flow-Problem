#include "common.cuh"
#include "boundary_conditions.h"

std::vector<float> uv_velocity(float Re, const std::vector<float>& y, int xN, int yN, int zN, std::vector<float>& u_inlet,
                                float dx, float dy, float dz, std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, std::vector<float>& p);