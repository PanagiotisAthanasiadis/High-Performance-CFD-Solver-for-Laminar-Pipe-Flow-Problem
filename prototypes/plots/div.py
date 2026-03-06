import numpy as np
import matplotlib.pyplot as plt

# 1. Setup the grid
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# 2. Define the vector field components (Diverging from origin)
# U is the x-component, V is the y-component
U = X
V = Y

# 3. Calculate the magnitude for coloring
speed = np.sqrt(U**2 + V**2)

# 4. Create the plot
plt.figure(figsize=(8, 7))

# Plotting the vectors (Quiver)
# 'pivot' anchors the arrow; 'color' uses the magnitude for a heatmap effect
q = plt.quiver(X, Y, U, V, speed, cmap='jet', pivot='mid', scale=25)

# Add a colorbar to show vector intensity
plt.colorbar(q, label='Magnitude')

# Add labels and styling
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid(alpha=0.3)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.savefig('divergence_field.png', dpi=600, bbox_inches='tight')