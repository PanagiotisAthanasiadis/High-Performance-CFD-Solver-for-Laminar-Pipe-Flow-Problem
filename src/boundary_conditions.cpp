#include "boundary_conditions.h"

void boundary_conditions(const std::vector<float>& y, int xN, int yN, int zN, std::vector<float>& u_inlet,
                         std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, std::vector<float>& p) {
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;
    int totalSize = sizeX * sizeY * sizeZ;

    u.assign(totalSize, 0.0);
    v.assign(totalSize, 0.0);
    w.assign(totalSize, 0.0);
    p.assign(totalSize, 0.0);

    // --- Fill interior from y[] ---
    for (int i = 1; i <= xN; i++) {
        for (int j = 1; j <= yN; j++) {
            for (int k = 1; k <= zN; k++) {
                int pos = (i - 1) * yN * zN + (j - 1) * zN + (k - 1);

                u[idx_3d(i, j, k, sizeY, sizeZ)] = y[pos];
                v[idx_3d(i, j, k, sizeY, sizeZ)] = y[xN * yN * zN + pos];
                w[idx_3d(i, j, k, sizeY, sizeZ)] = y[2 * xN * yN * zN + pos];
                p[idx_3d(i, j, k, sizeY, sizeZ)] = y[3 * xN * yN * zN + pos];
            }
        }
    }

    // --- Boundary conditions ---

    // X-direction boundaries
    for (int j = 0; j < sizeY; j++) {
        for (int k = 0; k < sizeZ; k++) {
            u[idx_3d(0, j, k, sizeY, sizeZ)]      = u_inlet[k * sizeY + j];          // inlet
            u[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = u[idx_3d(xN, j, k, sizeY, sizeZ)]; // outlet
            v[idx_3d(0, j, k, sizeY, sizeZ)]      = 0.0;
            v[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = v[idx_3d(xN, j, k, sizeY, sizeZ)];
            w[idx_3d(0, j, k, sizeY, sizeZ)]      = 0.0;
            w[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = w[idx_3d(xN, j, k, sizeY, sizeZ)];
            p[idx_3d(0, j, k, sizeY, sizeZ)]      = p[idx_3d(1, j, k, sizeY, sizeZ)];
            p[idx_3d(xN + 1, j, k, sizeY, sizeZ)] = 0.0;
        }
    }

    // Y-direction boundaries
    for (int i = 0; i < sizeX; i++) {
        for (int k = 0; k < sizeZ; k++) {
            u[idx_3d(i, 0, k, sizeY, sizeZ)]      = 0.0;
            u[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = 0.0;
            v[idx_3d(i, 0, k, sizeY, sizeZ)]      = 0.0;
            v[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = 0.0;
            w[idx_3d(i, 0, k, sizeY, sizeZ)]      = 0.0;
            w[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = 0.0;
            p[idx_3d(i, 0, k, sizeY, sizeZ)]      = p[idx_3d(i, 1, k, sizeY, sizeZ)];
            p[idx_3d(i, yN + 1, k, sizeY, sizeZ)] = p[idx_3d(i, yN, k, sizeY, sizeZ)];
        }
    }

    // Z-direction boundaries
    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            u[idx_3d(i, j, 0, sizeY, sizeZ)]      = 0.0;
            u[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = 0.0;
            v[idx_3d(i, j, 0, sizeY, sizeZ)]      = 0.0;
            v[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = 0.0;
            w[idx_3d(i, j, 0, sizeY, sizeZ)]      = 0.0;
            w[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = 0.0;
            p[idx_3d(i, j, 0, sizeY, sizeZ)]      = p[idx_3d(i, j, 1, sizeY, sizeZ)];
            p[idx_3d(i, j, zN + 1, sizeY, sizeZ)] = p[idx_3d(i, j, zN, sizeY, sizeZ)];
        }
    }
}