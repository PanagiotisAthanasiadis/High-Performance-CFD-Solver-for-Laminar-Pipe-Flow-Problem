#include "common.cuh"

void boundary_conditions(const std::vector<float>& y, int xN, int yN, int zN, std::vector<float>& u_inlet,
                         std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, std::vector<float>& p);