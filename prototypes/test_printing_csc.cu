#include <iostream>
#include <vector>
#include <iomanip>

// ---------------------------------------------------------
// Function: Print CSC Matrix (Host Side)
// ---------------------------------------------------------
void print_csc_matrix(
    int num_rows,           // Number of rows
    int num_cols,           // Number of columns
    const int* h_col_ptr,   // Column Pointers (size: num_cols + 1)
    const int* h_row_ind,   // Row Indices (size: nnz)
    const float* h_val      // Values (size: nnz)
) {
    printf("--- CSC Matrix Dump (%d x %d) ---\n", num_rows, num_cols);
    printf("Format: (Row, Col) : Value\n");
    printf("-----------------------------\n");

    // CSC iterates Column by Column
    for (int col = 0; col < num_cols; col++) {
        
        // The range of indices for this column
        int start_idx = h_col_ptr[col];
        int end_idx   = h_col_ptr[col + 1];

        printf("Column %d (Indices %d to %d):\n", col, start_idx, end_idx);

        if (start_idx == end_idx) {
            printf("  (Empty)\n");
        }

        for (int i = start_idx; i < end_idx; i++) {
            int row = h_row_ind[i];
            float val = h_val[i];

            // Print the triplet
            printf("  (%d, %d) : %.4f", row, col, val);

            // Validation Check (Helpful for debugging)
            if (row < 0 || row >= num_rows) {
                printf("  <-- ERROR: Row Index Out of Bounds! (0 to %d)", num_rows - 1);
            }
            printf("\n");
        }
    }
    printf("-----------------------------\n");
    printf("Total NNZ: %d\n", h_col_ptr[num_cols]);
}

// ---------------------------------------------------------
// Example Usage
// ---------------------------------------------------------
int main() {
    // Example Matrix (3 rows, 3 cols)
    //     Col 0   Col 1   Col 2
    // R0 [  4       1       0  ]
    // R1 [  1       4       1  ]
    // R2 [  0       1       4  ]
    
    // CSC Data (Column-Major)
    // Col 0: Rows [0, 1], Vals [4, 1]
    // Col 1: Rows [0, 1, 2], Vals [1, 4, 1]
    // Col 2: Rows [1, 2], Vals [1, 4]

    int num_rows = 3;
    int num_cols = 3;
    
    // CSC Arrays
    std::vector<int>   h_col_ptr = { 0, 2, 5, 7 }; // Size = cols + 1
    std::vector<int>   h_row_ind = { 0, 1, 0, 1, 2, 1, 2 };
    std::vector<float> h_val     = { 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0 };

    print_csc_matrix(num_rows, num_cols, 
                     h_col_ptr.data(), 
                     h_row_ind.data(), 
                     h_val.data());

    return 0;
}