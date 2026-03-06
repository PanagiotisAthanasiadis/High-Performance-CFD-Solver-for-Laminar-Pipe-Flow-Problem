import numpy as np
import matplotlib.pyplot as plt

# 1. Setup the grid
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# 2. Define the vector field components (Counter-clockwise rotation)
# U = -y, V = x
U = -Y
V = X

# 3. Calculate magnitude for color mapping
speed = np.sqrt(U**2 + V**2)

# 4. Create the plot
plt.figure(figsize=(8, 7))

# Quiver plot for the vectors
# We use the 'magma' colormap to contrast with the previous divergence plot
q = plt.quiver(X, Y, U, V, speed, cmap='jet', pivot='mid')

# Add a colorbar
plt.colorbar(q, label='Magnitude')

# 5. Add Streamlines to highlight the "swirl"
plt.streamplot(X, Y, U, V, color='gray', linewidth=0.5, density=0.8)

# Styling
plt.title('2D Curl Vector Field')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(alpha=0.2)

# 6. Export to PNG
plt.savefig('curl_field_plot.png', dpi=600, bbox_inches='tight')
print("Plot saved as curl_field_plot.png")

plt.show()