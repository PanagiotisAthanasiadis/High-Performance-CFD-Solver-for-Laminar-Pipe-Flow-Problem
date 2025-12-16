#include "common.cuh"
#include "solver.cuh"


// ============================================================================
// Convert COO to CSR on GPU 
// ============================================================================

void convert_coo_to_csr_gpu(
    int* d_row_coo, int* d_col_coo, float* d_vals_coo, int nnz, int nCell,
    int** d_row_ptr, int** d_col_idx, float** d_values)
{
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Allocate CSR arrays
    CUDA_CHECK(cudaMalloc(d_row_ptr, (nCell + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(d_values, nnz * sizeof(float)));
    
    // Convert COO to CSR
    cusparseXcoo2csr(handle, d_row_coo, nnz, nCell,
                     *d_row_ptr, CUSPARSE_INDEX_BASE_ZERO);
    
    CUDA_CHECK(cudaMemcpy(*d_col_idx, d_col_coo, nnz * sizeof(int), 
                         cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(*d_values, d_vals_coo, nnz * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    cusparseDestroy(handle);
}

void solve_with_cusolver_sparse(
    int* d_row_ptr, int* d_col_idx, float* d_vals, int nnz,
    float* d_residuals, float* d_solution, int nCell)
{
    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);
    
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    
    // Use cusolverSpScsrlsvqr for sparse solve
    float tol = 1e-6f;
    int singularity;
    
    cusolverSpScsrlsvqr(handle, nCell, nnz, descrA,
                       d_vals, d_row_ptr, d_col_idx,
                       d_residuals, tol, 0, d_solution, &singularity);
    
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}
