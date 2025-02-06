import numpy as np
import pandas as pd

# Define the x and y limits
x_min, x_max = 2.0, 12.0  # Change as needed
y_min, y_max = -1.5, 1.5  # Change as needed

# Define the number of points in x and y directions
num_x = 6  # Change as needed
num_y = 3  # Change as needed

# Generate a grid of x and y values
x_vals = np.linspace(x_min, x_max, num_x)
y_vals = np.linspace(y_min, y_max, num_y)

# Create coordinate list (z is assumed to be 0)
coords = [(x, y, 0.0) for x in x_vals for y in y_vals]

# Convert to DataFrame and save as CSV
df = pd.DataFrame(coords, columns=["x", "y", "z"])
df.to_csv("inputs/measurement_coords.csv", index=False)

print("measurement_coords.csv has been generated successfully!")
print(df)

