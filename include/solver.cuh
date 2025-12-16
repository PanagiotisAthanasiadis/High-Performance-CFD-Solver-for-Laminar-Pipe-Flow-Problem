#pragma once

void convert_coo_to_csr_gpu(
    int* d_row_coo, int* d_col_coo, float* d_vals_coo, int nnz, int nCell,
    int** d_row_ptr, int** d_col_idx, float** d_values);

void solve_with_cusolver_sparse(
    int* d_row_ptr, int* d_col_idx, float* d_vals, int nnz,
    float* d_residuals, float* d_solution, int nCell);
