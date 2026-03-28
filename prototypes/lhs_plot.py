import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# 1. Load the archive
data = np.load('lhs.npz')

# Extract arrays
vals = data['values']
cols = data['col_indices']
rows = data['row_pointers']
shape = tuple(data['shape'])

# 2. Reconstruct the SciPy sparse matrix
lhs = csr_matrix((vals, cols, rows), shape=shape)

# Convert to COOrdinate format to easily get (x, y, value) for scatter plotting
lhs_coo = lhs.tocoo()

# 3. Visualize
plt.figure(figsize=(10, 8))

# Scatter plot: x=columns, y=rows, c=values (colors)
# s=1 is the marker size. cmap='coolwarm' is great for seeing positive/negative swings
scatter = plt.scatter(lhs_coo.col, lhs_coo.row, 
                      c=lhs_coo.data, cmap='coolwarm', s=2, alpha=0.8)

# Invert Y axis so row 0 is at the top (standard matrix visual representation)
plt.gca().invert_yaxis()

plt.colorbar(scatter, label="lhs Value ")
plt.title(f"CFD lhs Matrix\nSize: {shape[0]}x{shape[1]}, NNZ: {len(vals)}")
plt.xlabel("Column Index")
plt.ylabel("Row Index")

plt.tight_layout()
plt.show()
plt.savefig('lhs_debug.png', dpi=600)

data.close()