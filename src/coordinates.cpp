#include "common.cuh"
#include "coordinates.h"

std::tuple<float, float, float>
coordinates(std::vector<float> &xcoor, std::vector<float> &ycoor,
            std::vector<float> &zcoor, const int xN, const int yN,
            const int zN, const float L, const float M, const float N) {
    int xSize = xN + 2;
    int ySize = yN + 2;
    int zSize = zN + 2;

    for (int iz = 0; iz < zSize; ++iz) {
        for (int iy = 0; iy < ySize; ++iy) {
            for (int ix = 0; ix < xSize; ++ix) {
                int idx = idx_3d(ix, iy, iz, ySize, zSize);
                xcoor[idx] = L * ix / (xSize - 1);
                ycoor[idx] = M * iy / (ySize - 1);
                zcoor[idx] = N * iz / (zSize - 1);
            }
        }
    }

    float dx = L / (xSize - 1);
    float dy = M / (ySize - 1);
    float dz = N / (zSize - 1);

    return std::make_tuple(dx, dy, dz);
}
