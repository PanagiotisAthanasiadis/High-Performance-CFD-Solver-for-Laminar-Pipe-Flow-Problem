import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Grid setup
x = y = z = np.linspace(-2, 2, 8)
X, Y, Z = np.meshgrid(x, y, z)

# Diverging components: F = [x, y, z]
U, V, W = X, Y, Z

# Calculate magnitude for coloring
mag = np.sqrt(U**2 + V**2 + W**2)

# Plotting with 'viridis'
# .flatten() is required for the color array in 3D quiver
q = ax.quiver(X, Y, Z, U, V, W, length=0.3, cmap='jet', array=mag.flatten(), normalize=True)

fig.colorbar(q, ax=ax, shrink=0.5, label='Magnitude')

plt.savefig('diverge_3d_colored.png', dpi=600)
plt.show()