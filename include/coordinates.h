#pragma once

#include <vector>
#include <tuple>

std::tuple<float, float, float>
coordinates(std::vector<float> &xcoor, std::vector<float> &ycoor,
            std::vector<float> &zcoor, const int xN, const int yN,
            const int zN, const float L, const float M, const float N);
