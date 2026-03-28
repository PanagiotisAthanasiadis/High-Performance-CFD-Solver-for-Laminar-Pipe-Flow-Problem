import numpy as np
import matplotlib.pyplot as plt
import collections

# 1. Load the archive
data = np.load('jacobian_coo.npz')

# Extract arrays
vals = data['values']
rows = data['row_indices']
cols = data['col_indices']
shape = tuple(data['shape'])

# ==========================================
# 2. ANALYSIS: FIND ARTIFACTS
# ==========================================
print("--- Jacobian Artifact Analysis ---")

# Count how many non-zeros are in each row and column
row_counts = collections.Counter(rows)
col_counts = collections.Counter(cols)

# Threshold for a suspiciously dense row/col (adjust if your stencil is very large)
artifact_threshold = 50 

artifact_rows = [(r, count) for r, count in row_counts.items() if count > artifact_threshold]
artifact_cols = [(c, count) for c, count in col_counts.items() if count > artifact_threshold]

artifact_rows.sort(key=lambda x: x[1], reverse=True)
artifact_cols.sort(key=lambda x: x[1], reverse=True)

print(f"Found {len(artifact_rows)} artifact ROWS (> {artifact_threshold} non-zeros):")
for r, count in artifact_rows[:10]: 
    eq_type = ["U-mom", "V-mom", "W-mom", "Continuity/Pressure"][r % 4]
    print(f"  Row {r} ({eq_type} eq for Cell {r//4}): {count} non-zeros")

print(f"\nFound {len(artifact_cols)} artifact COLUMNS (> {artifact_threshold} non-zeros):")
for c, count in artifact_cols[:10]: 
    var_type = ["u", "v", "w", "p"][c % 4]
    print(f"  Col {c} ({var_type} var for Cell {c//4}): {count} non-zeros")

# ==========================================
# 3. FILTERING: ISOLATE PRESSURE
# ==========================================
# Assuming variables are ordered u, v, w, p -> p is at index 3, 7, 11, etc.
is_p_row = (rows % 4 == 3)
is_p_col = (cols % 4 == 3)

# Find entries where BOTH the row is a pressure equation AND the column is a pressure variable
is_p_to_p = is_p_row & is_p_col

# Split the data into "Normal" and "Pressure-to-Pressure" entries for plotting
norm_rows = rows[~is_p_to_p]
norm_cols = cols[~is_p_to_p]
norm_vals = vals[~is_p_to_p]

p_rows = rows[is_p_to_p]
p_cols = cols[is_p_to_p]
p_vals = vals[is_p_to_p]

# ==========================================
# 4. VISUALIZATION (SIDE-BY-SIDE)
# ==========================================
# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# --- PLOT 1: ORIGINAL ---
scatter1 = ax1.scatter(cols, rows, c=vals, cmap='coolwarm', s=2, alpha=1)
ax1.invert_yaxis()
cbar1 = fig.colorbar(scatter1, ax=ax1)
cbar1.set_label("Jacobian Value")
ax1.set_title(f"Original CFD Jacobian Matrix\nSize: {shape[0]}x{shape[1]}, NNZ: {len(vals)}")
ax1.set_xlabel("Column Index")
ax1.set_ylabel("Row Index")
ax1.set_xlim(0, shape[1])
ax1.set_ylim(shape[0], 0)

# --- PLOT 2: HIGHLIGHTED PRESSURE ---
# Plot standard velocities/momentum equations (semi-transparent)
scatter_norm = ax2.scatter(norm_cols, norm_rows, c=norm_vals, cmap='coolwarm', 
                           s=1, alpha=0.3, label='Momentum/Velocity')

# Plot pressure-to-pressure dependencies on top (opaque, distinct colormap)
scatter_p = ax2.scatter(p_cols, p_rows, c=p_vals, cmap='spring', 
                        s=5, alpha=1.0, label='Pressure-to-Pressure')
ax2.invert_yaxis()

# Add individual colorbars for the second plot
cbar_norm = fig.colorbar(scatter_norm, ax=ax2, fraction=0.046, pad=0.04)
cbar_norm.set_label("Velocity Jacobian Values", size=10)

cbar_p = fig.colorbar(scatter_p, ax=ax2, fraction=0.046, pad=0.08)
cbar_p.set_label("Pressure Jacobian Values", size=10)

ax2.set_title("CFD Jacobian with Pressure Highlighted")
ax2.set_xlabel("Column Index (Variables)")
ax2.set_ylabel("Row Index (Equations)")
ax2.set_xlim(0, shape[1])
ax2.set_ylim(shape[0], 0)
ax2.legend(loc='upper right', framealpha=0.9)

# Final formatting and save
plt.tight_layout()
plt.savefig('jacobian_debug_combined.png', dpi=600)
plt.show()

# Clean up
data.close()