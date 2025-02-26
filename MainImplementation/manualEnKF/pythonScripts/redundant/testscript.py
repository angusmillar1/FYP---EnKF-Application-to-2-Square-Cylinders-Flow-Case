import numpy as np
import pandas as pd

# Path to the CSV file containing probe point coordinates
probe_coords_file = "outputs/sample_points_locations.csv"

# Read probe coordinates from the CSV file
probe_coords = pd.read_csv(probe_coords_file, skiprows=1, header=None).values
print(probe_coords)