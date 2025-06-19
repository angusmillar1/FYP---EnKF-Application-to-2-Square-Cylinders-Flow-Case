# Not used anymore
# Create a zoomed in visualisation of the measurement points selected
# Saved to outputs/ and displayed at the start of the run to verify selected points are as expected

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import time

# Define domain limits for plot
x_range = [-4, 12]
y_range = [-4, 4]

# Read in point coords
probe_coords_file = "inputs/measurement_coords.csv"
probe_coords = pd.read_csv(probe_coords_file, skiprows=1, header=None).values
print(probe_coords)

# Create plots
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the original domain for reference
ax.set_xlim(x_range)
ax.set_ylim(y_range)

# Plot the reduced resolution grid (probed points)
ax.scatter(probe_coords[:, 1], probe_coords[:, 2], color="blue", s=10)

# Draw squares for context
square1 = patches.Rectangle((-0.5, 0.5), 1, 1, linewidth=1, edgecolor="black", facecolor="black")
square2 = patches.Rectangle((-0.5, -1.5), 1, 1, linewidth=1, edgecolor="black", facecolor="black")
ax.add_patch(square1)
ax.add_patch(square2)

# Add labels, legend, and grid
ax.set_title("Reduced Resolution Mesh")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid()
ax.set_aspect('equal')
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

# Save the plot
plt.savefig("outputs/reduced_mesh_vis_zoom.png", dpi=300)

# Display plot, but do not block execution and close after 5 seconds
plt.show(block=False)
start_plot_timer = time.time()
while time.time() - start_plot_timer < 3:
    plt.pause(0.1)  # Allow GUI updates during the sleep
plt.close()

