#include "uv_velocity.h"

std::vector<float> uv_velocity(float Re, const std::vector<float>& y, int xN, int yN, int zN, std::vector<float>& u_inlet,
                                float dx, float dy, float dz, std::vector<float>& u, std::vector<float>& v, std::vector<float>& w, std::vector<float>& p) {
    
    boundary_conditions(y, xN, yN, zN, u_inlet, u, v, w, p);
    
    // Sizes including ghost cells
    int sizeX = xN + 2;
    int sizeY = yN + 2;
    int sizeZ = zN + 2;

    int nCell = xN * yN * zN;
    std::vector<float> out(4 * nCell, 0.0);

    // interior loop (i=1..xN, j=1..yN, k=1..zN)
    for (int i = 1; i <= xN; ++i) {
        for (int j = 1; j <= yN; ++j) {
            for (int k = 1; k <= zN; ++k) {

                int pos = (i - 1) * (yN * zN) + (j - 1) * zN + (k - 1);

                // U-momentum (first block)
                {
                    float conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*u[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*u[idx_3d(i-1,j,k,sizeY,sizeZ)] );
                    float conv_y = 0.5 * dx * dz * ( u[idx_3d(i,j+1,k,sizeY,sizeZ)]*v[idx_3d(i,j+1,k,sizeY,sizeZ)] - u[idx_3d(i,j-1,k,sizeY,sizeZ)]*v[idx_3d(i,j-1,k,sizeY,sizeZ)] );
                    float conv_z = 0.5 * dx * dy * ( u[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - u[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
                    float pres    = (dy * dz) * ( p[idx_3d(i+1,j,k,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

                    float diff = (1.0/Re) * (
                        (dy*dz/dx) * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
                        (dx*dz/dy) * ( u[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
                        (dx*dy/dz) * ( u[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*u[idx_3d(i,j,k,sizeY,sizeZ)] + u[idx_3d(i,j,k-1,sizeY,sizeZ)] )
                    );

                    out[ pos ] = conv_x + conv_y + conv_z + pres - diff;
                }

                // V-momentum (second block)
                {
                    float conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*v[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*v[idx_3d(i-1,j,k,sizeY,sizeZ)] );
                    float conv_y = 0.5 * dx * dz * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)]*v[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)]*v[idx_3d(i,j-1,k,sizeY,sizeZ)] );
                    float conv_z = 0.5 * dx * dy * ( v[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - v[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
                    float pres    = (dx * dz) * ( p[idx_3d(i,j+1,k,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

                    float diff = (1.0/Re) * (
                        (dy*dz/dx) * ( v[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
                        (dx*dz/dy) * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
                        (dx*dy/dz) * ( v[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*v[idx_3d(i,j,k,sizeY,sizeZ)] + v[idx_3d(i,j,k-1,sizeY,sizeZ)] )
                    );

                    out[ nCell + pos ] = conv_x + conv_y + conv_z + pres - diff;
                }

                // W-momentum (third block)
                {
                    float conv_x = 0.5 * dy * dz * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)]*w[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)]*w[idx_3d(i-1,j,k,sizeY,sizeZ)] );
                    float conv_y = 0.5 * dx * dz * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)]*w[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)]*w[idx_3d(i,j-1,k,sizeY,sizeZ)] );
                    float conv_z = 0.5 * dx * dy * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)]*w[idx_3d(i,j,k+1,sizeY,sizeZ)] - w[idx_3d(i,j,k-1,sizeY,sizeZ)]*w[idx_3d(i,j,k-1,sizeY,sizeZ)] );
                    float pres    = (dx * dy) * ( p[idx_3d(i,j,k+1,sizeY,sizeZ)] - p[idx_3d(i,j,k,sizeY,sizeZ)] );

                    float diff = (1.0/Re) * (
                        (dy*dz/dx) * ( w[idx_3d(i+1,j,k,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i-1,j,k,sizeY,sizeZ)] ) +
                        (dx*dz/dy) * ( w[idx_3d(i,j+1,k,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i,j-1,k,sizeY,sizeZ)] ) +
                        (dx*dy/dz) * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)] - 2.0*w[idx_3d(i,j,k,sizeY,sizeZ)] + w[idx_3d(i,j,k-1,sizeY,sizeZ)] )
                    );

                    out[ 2*nCell + pos ] = conv_x + conv_y + conv_z + pres - diff;
                }

                // Continuity (fourth block)
                {
                    float cont = (dy*dz/2.0) * ( u[idx_3d(i+1,j,k,sizeY,sizeZ)] - u[idx_3d(i-1,j,k,sizeY,sizeZ)] )
                                + (dx*dz/2.0) * ( v[idx_3d(i,j+1,k,sizeY,sizeZ)] - v[idx_3d(i,j-1,k,sizeY,sizeZ)] )
                                + (dx*dy/2.0) * ( w[idx_3d(i,j,k+1,sizeY,sizeZ)] - w[idx_3d(i,j,k-1,sizeY,sizeZ)] );

                    out[ 3*nCell + pos ] = cont;
                }
            }
        }
    }

    return out;
}