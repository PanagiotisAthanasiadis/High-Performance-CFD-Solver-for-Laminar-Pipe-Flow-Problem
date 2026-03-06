import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Grid setup
x = y = z = np.linspace(-2, 2, 10)
X, Y, Z = np.meshgrid(x, y, z)

# Curl components (Rotating around Z-axis): F = [-y, x, 0]
U = -Y
V = X
W = np.zeros_like(Z)

# Calculate magnitude (rotational speed)
mag = np.sqrt(U**2 + V**2 + W**2)

# Plotting with 'magma'
q = ax.quiver(X, Y, Z, U, V, W, length=0.3, cmap='jet', array=mag.flatten(), normalize=True)

fig.colorbar(q, ax=ax, shrink=0.5, label='Magnitude')
plt.savefig('curl_3d_colored.png', dpi=600)
plt.show()