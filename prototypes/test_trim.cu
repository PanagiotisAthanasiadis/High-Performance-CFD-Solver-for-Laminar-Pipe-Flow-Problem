#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// -------------------------------------------------------------------------
// Error Handling Macros
// -------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel Launch Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } wh


// -------------------------------------------------------------------------
// Kernel: Generate Selection Flags
// -------------------------------------------------------------------------
__global__ void generate_flags_kernel(int nnz, const float* values, char* flags, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        // 1 = Keep, 0 = Discard
        flags[idx] = (fabsf(values[idx]) > threshold) ? 1 : 0;
    }
}

// -------------------------------------------------------------------------
// Main Filter Function (CSR Format using CUB)
// -------------------------------------------------------------------------
void filter_csr_cub(int rows, int old_nnz,
                    int* d_row_offsets, int* d_cols, float* d_vals,
                    int** d_new_row_offsets, int** d_new_cols, float** d_new_vals,
                    int* new_nnz_out,float threshold) {
    
    // ---------------------------------------------------------------------
    // 1. Setup & Generate Flags
    // ---------------------------------------------------------------------
    char* d_flags;
    CUDA_CHECK(cudaMalloc((void**)&d_flags, old_nnz * sizeof(char)));

    int blockSize = 256;
    int gridSize = (old_nnz + blockSize - 1) / blockSize;
    
    generate_flags_kernel<<<gridSize, blockSize>>>(old_nnz, d_vals, d_flags, threshold);

    // ---------------------------------------------------------------------
    // 2. Calculate New Row Counts (Segmented Reduce)
    // ---------------------------------------------------------------------
    // We sum the 'flags' within each row segment to see how many kept items per row.
    
    int* d_row_counts; 
    CUDA_CHECK(cudaMalloc((void**)&d_row_counts, rows * sizeof(int)));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // A. Query memory requirement
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                               d_flags, d_row_counts, 
                                               rows, d_row_offsets, d_row_offsets + 1));
    
    // B. Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // C. Run Reduction
    CUDA_CHECK(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, 
                                               d_flags, d_row_counts, 
                                               rows, d_row_offsets, d_row_offsets + 1));

    // ---------------------------------------------------------------------
    // 3. Generate New Row Offsets (Prefix Scan)
    // ---------------------------------------------------------------------
    
    // Allocate Output Row Pointer Array (Size: Rows + 1)
    CUDA_CHECK(cudaMalloc((void**)d_new_row_offsets, (rows + 1) * sizeof(int)));
    
    // Cleanup previous temp storage to be safe (or reuse if you manage size manually)
    CUDA_CHECK(cudaFree(d_temp_storage)); 
    d_temp_storage = NULL; 
    temp_storage_bytes = 0;
    
    // A. Query memory requirement
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                             d_row_counts, *d_new_row_offsets, rows));
    
    // B. Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // C. Run Scan
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                             d_row_counts, *d_new_row_offsets, rows));

    // Calculate total NNZ (Read last count and last offset from GPU)
    int last_count = 0, last_offset = 0;
    
    // Note: d_row_counts has size 'rows'. Last index is rows-1.
    //       d_new_row_offsets has size 'rows+1'. Index 'rows' is currently empty/garbage.
    
    CUDA_CHECK(cudaMemcpy(&last_count, &d_row_counts[rows-1], sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_offset, *d_new_row_offsets + (rows-1), sizeof(int), cudaMemcpyDeviceToHost));
    
    int final_nnz = last_offset + last_count;
    *new_nnz_out = final_nnz;
    
    // Write the final NNZ to the end of the row pointer array (Index 'rows')
    CUDA_CHECK(cudaMemcpy(*d_new_row_offsets + rows, &final_nnz, sizeof(int), cudaMemcpyHostToDevice));

    printf("Filtering Result: %d -> %d elements\n", old_nnz, final_nnz);

    // ---------------------------------------------------------------------
    // 4. Compact Data Arrays (DeviceSelect)
    // ---------------------------------------------------------------------
    
    // Allocate Result Arrays
    CUDA_CHECK(cudaMalloc((void**)d_new_cols, final_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)d_new_vals, final_nnz * sizeof(float)));
    
    // CUB requires a pointer to store the number of selected items (on device)
    int* d_num_selected;
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected, sizeof(int)));

    // Cleanup temp storage again
    CUDA_CHECK(cudaFree(d_temp_storage)); 
    d_temp_storage = NULL; 
    temp_storage_bytes = 0;
    
    // A. Query memory requirement (Using d_vals as reference)
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_vals, d_flags, *d_new_vals, d_num_selected, old_nnz));
    
    // B. Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // C. Run Select (Compact Values)
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_vals, d_flags, *d_new_vals, d_num_selected, old_nnz));

    // D. Run Select (Compact Columns)
    // Note: We can reuse d_temp_storage because the operation size is identical
    CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, 
                                          d_cols, d_flags, *d_new_cols, d_num_selected, old_nnz));

    // Verify correct number of items selected (Optional Check)
    int debug_nnz = 0;
    CUDA_CHECK(cudaMemcpy(&debug_nnz, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost));
    if (debug_nnz != final_nnz) {
        fprintf(stderr, "Mismatch! Calculated NNZ: %d, DeviceSelect NNZ: %d\n", final_nnz, debug_nnz);
    }

    // ---------------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_row_counts));
    CUDA_CHECK(cudaFree(d_num_selected));
    
    // Wait for everything to finish
    CUDA_CHECK(cudaDeviceSynchronize());
}


// -------------------------------------------------------------------------
// Test Case: Drop Everything
// -------------------------------------------------------------------------
int main() {
    // 1. Setup Parameters
    const int ROWS = 10000;
    const int NNZ_PER_ROW = 1000;
    const int INPUT_NNZ = ROWS * NNZ_PER_ROW; // 1 Million Elements
    
    // 2. Define Thresholds
    // We will generate values = 0.2f
    // We will set threshold   = 0.5f
    // EXPECTATION: 0.2 < 0.5 --> DROP ALL
    float val_to_generate = 0.2f;
    float threshold       = 0.2f;

    printf("--- Test: Filter to Zero ---\n");
    printf("Input NNZ: %d\n", INPUT_NNZ);
    printf("Values:    %f\n", val_to_generate);
    printf("Threshold: %f\n", threshold);

    // 3. Generate Host Data (Pinned memory for speed)
    int *h_row_ptr, *h_cols;
    float *h_vals;
    
    cudaMallocHost((void**)&h_row_ptr, (ROWS + 1) * sizeof(int));
    cudaMallocHost((void**)&h_cols, INPUT_NNZ * sizeof(int));
    cudaMallocHost((void**)&h_vals, INPUT_NNZ * sizeof(float));

    // Fill Data
    h_row_ptr[0] = 0;
    for (int i = 0; i < ROWS; i++) {
        h_row_ptr[i+1] = h_row_ptr[i] + NNZ_PER_ROW;
    }
    
    for (int i = 0; i < INPUT_NNZ; i++) {
        h_cols[i] = i % ROWS;       // Dummy column index
        h_vals[i] = val_to_generate; // ALL VALUES ARE 0.2
    }

    // 4. Move to GPU
    int *d_row_ptr, *d_cols;
    float *d_vals;
    
    CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (ROWS + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cols, INPUT_NNZ * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_vals, INPUT_NNZ * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols, INPUT_NNZ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, h_vals, INPUT_NNZ * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Run Filter
    int *d_new_row_ptr, *d_new_cols;
    float *d_new_vals;
    int final_nnz = -1; // Initialize to -1 to ensure it changes

    printf("Running Filter...\n");
    
    // ASSUMING YOUR FUNCTION SIGNATURE:
    filter_csr_cub(ROWS, INPUT_NNZ, 
                   d_row_ptr, d_cols, d_vals, 
                   &d_new_row_ptr, &d_new_cols, &d_new_vals, 
                   &final_nnz,
                   threshold); 
    
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Verification
    printf("Filter Complete.\n");
    printf("Final NNZ: %d\n", final_nnz);

    // CHECK 1: NNZ should be 0
    if (final_nnz == 0) {
        printf("[SUCCESS] NNZ is 0 as expected.\n");
    } else {
        printf("[FAILURE] NNZ is %d (Expected 0).\n", final_nnz);
        return 1;
    }

    // CHECK 2: Row Pointers should all be 0
    // Because if 0 items remain, the cumulative sum (scan) of counts is 0, 0, 0...
    int* h_check_rows = (int*)malloc((ROWS + 1) * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_check_rows, d_new_row_ptr, (ROWS + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    
    bool rows_ok = true;
    for(int i=0; i<=ROWS; i++) {
        if(h_check_rows[i] != 0) {
            printf("[FAILURE] Row offset %d is %d (Expected 0)\n", i, h_check_rows[i]);
            rows_ok = false;
            break;
        }
    }
    if(rows_ok) printf("[SUCCESS] All row offsets are 0.\n");

    // Cleanup
    cudaFreeHost(h_row_ptr); cudaFreeHost(h_cols); cudaFreeHost(h_vals);
    cudaFree(d_row_ptr); cudaFree(d_cols); cudaFree(d_vals);
    cudaFree(d_new_row_ptr); cudaFree(d_new_cols); cudaFree(d_new_vals);
    free(h_check_rows);

    return 0;
}