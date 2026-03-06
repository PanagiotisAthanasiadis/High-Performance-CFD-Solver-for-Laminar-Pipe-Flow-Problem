import numpy as np
import matplotlib.pyplot as plt

data = np.load("simulation_results.npz")

u=data['u']
v=data['v']
w=data['w']

p =data['p']
velmag=data['velmag']

xcoor=data['xcoor']
ycoor=data['ycoor']
zcoor=data['zcoor']

xN=data['xN'][0]
yN=data['yN'][0]
zN=data['zN'][0]

L=data['L'][0]
M=data['M'][0]
N=data['N'][0]

def plot_scalar_field(x, y, z, scalar, title):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(),
                    c=scalar.flatten(), cmap='jet', s=2)
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")

# --- 3D Plots ---
plot_scalar_field(xcoor[1:-1, 1:-1, 1:-1], ycoor[1:-1, 1:-1, 1:-1], zcoor[1:-1, 1:-1, 1:-1],
                  u[1:-1, 1:-1, 1:-1], "Velocity U Component")

plot_scalar_field(xcoor[1:-1, 1:-1, 1:-1], ycoor[1:-1, 1:-1, 1:-1], zcoor[1:-1, 1:-1, 1:-1],
                  p[1:-1, 1:-1, 1:-1], "Pressure Field")

# --- 2D Slices ---
ix, iy, iz = xN // 2, yN // 2, zN // 2

# --- Figure 1: YZ Slice ---
plt.figure()
plt.contourf(ycoor[ix, :, :], zcoor[ix, :, :], u[ix, :, :], 100, cmap='jet')
plt.colorbar()
plt.title("YZ Slice - U Velocity")
plt.xlabel("Y")
plt.ylabel("Z")
# Save immediately before opening the next figure
plt.savefig("YZ_slice.png", dpi=300, bbox_inches='tight') 

# --- Figure 2: XZ Slice ---
plt.figure()
plt.contourf(xcoor[:, iy, :], zcoor[:, iy, :], u[:, iy, :], 100, cmap='jet')
plt.colorbar()
plt.title("XZ Slice - U Velocity")
plt.xlabel("X")
plt.ylabel("Z")
# Save this specific figure
plt.savefig("XZ_slice.png", dpi=300, bbox_inches='tight')

# --- Figure 3: XY Slice ---
plt.figure()
plt.contourf(xcoor[:, :, iz], ycoor[:, :, iz], u[:, :, iz], 100, cmap='jet')
plt.colorbar()
plt.title("XY Slice - U Velocity")
plt.xlabel("X")
plt.ylabel("Y")
# Save this specific figure (I renamed it from result.png for consistency)
plt.savefig("XY_slice.png", dpi=300, bbox_inches='tight')

# --- Show all figures at once on the screen ---
#plt.show(block=True)

# 1. Create a single master figure with 1 row and 3 columns
# figsize=(18, 5) makes the image wide enough to fit three plots side-by-side
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# --- First Plot (Left): YZ Slice on axes[0] ---
# Note: We save the contourf output to 'c1' so we can attach a colorbar to it
c1 = axes[0].contourf(ycoor[ix, :, :], zcoor[ix, :, :], u[ix, :, :], 100, cmap='jet')
fig.colorbar(c1, ax=axes[0]) # Attach colorbar specifically to this subplot
axes[0].set_title("YZ Slice - U Velocity")
axes[0].set_xlabel("Y")
axes[0].set_ylabel("Z")

# --- Second Plot (Middle): XZ Slice on axes[1] ---
c2 = axes[1].contourf(xcoor[:, iy, :], zcoor[:, iy, :], u[:, iy, :], 100, cmap='jet')
fig.colorbar(c2, ax=axes[1])
axes[1].set_title("XZ Slice - U Velocity")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Z")

# --- Third Plot (Right): XY Slice on axes[2] ---
c3 = axes[2].contourf(xcoor[:, :, iz], ycoor[:, :, iz], u[:, :, iz], 100, cmap='jet')
fig.colorbar(c3, ax=axes[2])
axes[2].set_title("XY Slice - U Velocity")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Y")

# 2. Adjust the layout
# This mathematically spaces the subplots so titles and colorbars don't overlap!
plt.tight_layout()

# 3. Save the single master figure
plt.savefig("all_velocity_slices.png", dpi=300, bbox_inches='tight')



# Create a large single figure to hold all 6 plots
fig = plt.figure(figsize=(18, 20)) # Width=18, Height=20 for a 3x2 grid

# -------------------------------------------------------------------
# ROW 1: 3D Visualizations
# -------------------------------------------------------------------

# 1. Top-Left: 3D Velocity
ax1 = fig.add_subplot(3, 2, 1, projection='3d')
sc1 = ax1.scatter(xcoor.flatten(), ycoor.flatten(), zcoor.flatten(), 
                  c=u.flatten(), s=50, cmap='jet', edgecolors='none')
fig.colorbar(sc1, ax=ax1, label='Velocity (u)', shrink=0.7)
ax1.set_xlabel('Channel Length (x)')
ax1.set_title('3D Computational Grid: Velocity')
ax1.set_box_aspect([L,M,N]) 
ax1.view_init(elev=25, azim=35) 
ax1.grid(True)

# 2. Top-Right: 3D Pressure
ax2 = fig.add_subplot(3, 2, 2, projection='3d')
sc2 = ax2.scatter(xcoor.flatten(), ycoor.flatten(), zcoor.flatten(), 
                  c=p.flatten(), s=50, cmap='jet', edgecolors='none')
fig.colorbar(sc2, ax=ax2, label='Pressure (p)', shrink=0.7)
ax2.set_xlabel('Channel Length (x)')
ax2.set_title('3D Computational Grid: Pressure')
ax2.set_box_aspect([L,M,N])
ax2.view_init(elev=25, azim=35)
ax2.grid(True)

# -------------------------------------------------------------------
# ROW 2 & 3: 2D Slices
# -------------------------------------------------------------------

# 3. Middle-Left: INLET XY plane slice
ax3 = fig.add_subplot(3, 2, 3)
mesh1 = ax3.pcolormesh(xcoor[:, :, 1], ycoor[:, :, 1], velmag[:, :, 1], 
                       cmap='jet', shading='gouraud')
fig.colorbar(mesh1, ax=ax3)
ax3.set_title('Velocity Magnitude Profile at Inlet (XY plane)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_aspect('equal')
ax3.autoscale(tight=True)

# 4. Middle-Right: OUTLET XY plane slice
ax4 = fig.add_subplot(3, 2, 4)
mesh2 = ax4.pcolormesh(xcoor[:, :, -1], ycoor[:, :, -1], p[:, :, -1], 
                       cmap='jet', shading='gouraud')
fig.colorbar(mesh2, ax=ax4)
ax4.set_title('Pressure Distribution at Inlet (XY plane)') 
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_aspect('equal')
ax4.autoscale(tight=True)

# 5. Bottom-Left: OUTLET YZ plane slice
ax5 = fig.add_subplot(3, 2, 5)
mesh3 = ax5.pcolormesh(ycoor[-1, :, :], zcoor[-1, :, :], velmag[-1, :, :], 
                       cmap='jet', shading='gouraud')
fig.colorbar(mesh3, ax=ax5)
ax5.set_title('Velocity Magnitude Profile at Outlet (YZ plane)')
ax5.set_xlabel('y')
ax5.set_ylabel('z')
ax5.set_aspect('equal')
ax5.autoscale(tight=True)

# 6. Bottom-Right: INLET YZ plane slice
ax6 = fig.add_subplot(3, 2, 6)
mesh4 = ax6.pcolormesh(ycoor[0, :, :], zcoor[0, :, :], velmag[0, :, :], 
                       cmap='jet', shading='gouraud')
fig.colorbar(mesh4, ax=ax6)
ax6.set_title('Velocity Magnitude Profile at Inlet (YZ plane)')
ax6.set_xlabel('y')
ax6.set_ylabel('z')
ax6.set_aspect('equal')
ax6.autoscale(tight=True)

# -------------------------------------------------------------------
# Final Formatting and Export
# -------------------------------------------------------------------

# Adjust layout so subplots don't overlap
plt.tight_layout(pad=3.0) # pad adds a bit of extra breathing room between plots

# EXPORT: Save everything as one high-resolution file
fig.savefig('Combined_Simulation_Dashboard.png', dpi=300, bbox_inches='tight')


# -------------------------------------------------------------------
# Boundary Value Check (Verification of No-Slip Condition)
# -------------------------------------------------------------------
print("--- Boundary Velocity Check ---")
print("Expected value at walls is 0.0 (No-Slip Condition)")

# 1. Y-axis Boundaries (Bottom and Top walls)
y_min_wall_max_vel = np.max(velmag[:, 0, :])
y_max_wall_max_vel = np.max(velmag[:, -1, :])
print(f"Max velocity at Y_min boundary: {y_min_wall_max_vel:.8f}")
print(f"Max velocity at Y_max boundary: {y_max_wall_max_vel:.8f}")

# 2. Z-axis Boundaries (Side walls)
z_min_wall_max_vel = np.max(velmag[:, :, 0])
z_max_wall_max_vel = np.max(velmag[:, :, -1])
print(f"Max velocity at Z_min boundary: {z_min_wall_max_vel:.8f}")
print(f"Max velocity at Z_max boundary: {z_max_wall_max_vel:.8f}")

# Optional: Check overall minimum velocity in the entire domain
print(f"Global minimum velocity in domain: {np.min(velmag):.8f}")
print("-------------------------------")

print("--- Checking Mass Conservation (Divergence) ---")
# Assuming uniform grid spacing for a quick estimation. 
# Adjust dx, dy, dz based on your actual grid.
dx = xcoor[1,0,0] - xcoor[0,0,0]
dy = ycoor[0,1,0] - ycoor[0,0,0]
dz = zcoor[0,0,1] - zcoor[0,0,0]

# Calculate gradients (assuming u is indexed [x, y, z])
du_dx = np.gradient(u, dx, axis=0)
dv_dy = np.gradient(v, dy, axis=1)
dw_dz = np.gradient(w, dz, axis=2)

divergence = du_dx + dv_dy + dw_dz

print(f"Max absolute divergence: {np.max(np.abs(divergence)):.8f}")
print(f"Average divergence: {np.mean(np.abs(divergence)):.8f}")