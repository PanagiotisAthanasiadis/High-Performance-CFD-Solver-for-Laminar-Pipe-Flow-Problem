#ifndef LEV_H
#define LEV_H

#include <cstddef>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <cudss.h> 
#include <cublas_v2.h>
#include <cnpy.h>

// Function Declarations
void print_gpu_array(const float* d_array, int n);
void print_csr_matrix(int rows, int nnz, int* d_row_ptr, int* d_cols, float* d_vals);
void print_coo_matrix_gpu(int rows, int cols, int nnz, const int* d_rows, const int* d_cols, const float* d_vals, int max_print);
std::tuple<float*,float*,float*,float*> boundary_conditions_final(float *ysol, const int xN, const int yN, const int zN, const float *u_inlet);
void uv_velocity_single(float *out, float Re, float *y, int xN, int yN, int zN, const float *u_inlet, float dx, float dy, float dz);
std::pair<float*, std::tuple<int*, int*, float*, int>> Residuals_Sparse_Jacobian_finite_diff(float Re, float *y, int xN, int yN, int zN, const float *u_inlet, float dx, float dy, float dz);
std::tuple<float, float, float> coordinates(std::vector<float> &xcoor, std::vector<float> &ycoor, std::vector<float> &zcoor, int xN, int yN, int zN, float L, float M, float N);
void filter_csr_cub(float threshold, int rows, int old_nnz, int* d_row_offsets, int* d_cols, float* d_vals, int** d_new_row_offsets, int** d_new_cols, float** d_new_vals, int* new_nnz_out);
void print_csc_matrix(int num_rows, int num_cols, const int* d_col_ptr, const int* d_row_ind, const float* d_val);
void compute_AtA_debug(int* d_rows_coo, int* d_cols_coo, float* d_vals_coo, int nnz, int num_rows, int num_cols, int** d_result_rows, int** d_result_cols, float** d_result_vals, int* result_nnz, int** d_AT_cscOffsets, int** d_AT_columns, float** d_AT_values);
void create_identity_csr_and_scale(int N, float alpha, int* d_row_offsets, int* d_cols, float* d_vals);
void add_csr_cusparse(float delta, int m, int n, int nnzA, const int* d_A_row_offsets, const int* d_A_columns, const float* d_A_values, int nnzB, const int* d_B_row_offsets, const int* d_B_columns, const float* d_B_values, int* nnzC_out, int** d_C_row_offsets, int** d_C_columns, float** d_C_values);
void scale_and_multiply_on_gpu(int rows, int cols, int nnz, const int* d_col_offsets, const int* d_row_indices, const float* d_values, const float* d_x, float alpha, float** d_y_out);
void solve_system_gpu(long int n, long int nnz, const int* d_row_offsets, const int* d_col_indices, const float* d_values, const float* d_b, float** d_x_out);
void sort_coo_matrix_cusparse(int num_rows, int num_cols, int nnz, int* d_rows, int* d_cols, float* d_vals);
void sort_coo_cub(int* d_rows, int* d_cols, float* d_vals, int nnz);
float square_norm(cublasHandle_t handle, float * residual, int n);
float L2_norm_squared(cublasHandle_t handle, float * residual, int n);
float* compute_vel_mag(int n, float* d_u, float* d_v, float* d_w);
std::vector<float> gpu_to_vector(const float* d_array, int size);
void levenberg_marquardt_solver(float Re, std::vector<float>& y, const int xN, const int yN, const int zN, const float* u_inlet, const float dx, const float dy, const float dz, int max_iterations, float initial_lambda, float lambda_factor, float tolerance);

#endif // LEV_H
