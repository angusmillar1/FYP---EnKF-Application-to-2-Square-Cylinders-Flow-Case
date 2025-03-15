import vtk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import time
import pandas as pd

# ----------------------- USER DEFINED INPUTS -------------------------------
# Define the domain size
x_range = [-10, 40]
y_range = [-15, 15]

# Define method with which to create reference points
probe_type = "specified"  # "uniform" / "specified" / "combined"

# Uniform distribution controls
reduction_percentage = 0.1  # Reduce to x% of the original resolution (=1 => 1%)

# Specified distribution controls
# custom_coordinates = [
#     [2.0, 1.0, 0.0],   
#     [4.0, 1.0, 0.0],  
#     [6.0, 1.0, 0.0],
#     [8.0, 1.0, 0.0],
#     [10.0, 1.0, 0.0],
#     [2.0, -1.0, 0.0],   
#     [4.0, -1.0, 0.0],  
#     [6.0, -1.0, 0.0],
#     [8.0, -1.0, 0.0],
#     [10.0, -1.0, 0.0]
# ]
df = pd.read_csv("inputs/measurementPoints/measurement_coords.csv")
custom_coordinates = df[['x', 'y', 'z']].values.tolist()

# Specify whether to plot resolution of reduced mesh or not
display_reduced_mesh = 1  # 1 / 0
# ----------------------------------------------------------------------------






# Function to build vtk functions if pyvista not installed
def create_vtk_functions(vtk_file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    mesh = reader.GetOutput()

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    return mesh, locator

# Load the mesh
# vtk_file_path = "../memberRunFiles/member1/VTK/member1_0.vtk"
vtk_file_path = sys.argv[1]
output_filename = "outputs/reduced_mesh_cell_ids.csv"

# Clear any previous runs to rewrite
if os.path.exists(output_filename):
    os.remove(output_filename)  # Deletes the file
    # print(f"{output_filename} has been deleted.")
# else:
    # print(f"{output_filename} does not exist in the current directory.")



# Read mesh data into a variable (check dependencies)
mesh, locator = create_vtk_functions(vtk_file_path)






# REDUCE GRID BASED ON UNFORM DISRIBUTION

if probe_type == "uniform" or probe_type == "combined":

    # Total number of cells in the original mesh
    total_cells = mesh.GetNumberOfCells()

    # Calculate the number of points in the reduced grid
    reduced_cell_count = int((total_cells * reduction_percentage) / 100)

    # Ensure a minimum number of points
    reduced_cell_count = max(reduced_cell_count, 1)

    # Calculate approximate grid size (rows and columns)
    aspect_ratio = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])
    num_rows = int(np.sqrt(reduced_cell_count / aspect_ratio))
    num_cols = int(reduced_cell_count / num_rows)

    # Generate uniformly spaced reduced grid points
    x_coords = np.linspace(x_range[0], x_range[1], num_cols)
    y_coords = np.linspace(y_range[0], y_range[1], num_rows)
    reduced_grid_uniform = np.array([[x, y, 0.0] for x in x_coords for y in y_coords])


## REDUCE GRID BASED ON INDIVIDUALLY DEFINED POINTS
if probe_type == "specified" or probe_type == "combined":
    # Define specific coordinates for the reduced grid
    reduced_grid_custom = np.array(custom_coordinates)

# ----------------------------------------------------------------------------------
## DECIDE WHICH MESH TO USE OR COMBINE, *** ONLY ONE UNCOMMENTED ***, (this control moved to top)

if probe_type == "uniform":
    reduced_grid = reduced_grid_uniform
elif probe_type == "specified":
    reduced_grid = reduced_grid_custom
elif probe_type == "combined":
    reduced_grid = np.vstack([reduced_grid_uniform, reduced_grid_custom])
else:
    raise ValueError("Invalid reduced mesh type")

# ----------------------------------------------------------------------------------

# Remove any repeated values
reduced_grid = np.unique(reduced_grid, axis=0)

# Map reduced grid points to the nearest cell centroids
extracted_points = []
cell_ids = []  # Store the cell IDs here
for point in reduced_grid:
    # Find the index of the closest cell in the mesh
    cell_id_temp = vtk.reference(0)
    locator.FindClosestPoint(point, [0.0, 0.0, 0.0], cell_id_temp, vtk.reference(0), vtk.reference(0.0))
    closest_cell_id = int(cell_id_temp)  # Closest cell ID    
    cell_ids.append(closest_cell_id)  # Save the cell ID

    # Retrieve the cell using the cell ID
    closest_cell = mesh.GetCell(closest_cell_id)

    # Calculate the centroid of the cell
    points = closest_cell.GetPoints()
    num_points = points.GetNumberOfPoints()
    centroid = [0.0, 0.0, 0.0]  # Initialize centroid coordinates
    #   Sum coordinates of all points in the cell
    for i in range(num_points):
        pt = points.GetPoint(i)
        centroid[0] += pt[0]
        centroid[1] += pt[1]
        centroid[2] += pt[2]
    # Average the coordinates
    centroid = [x / num_points for x in centroid]

    extracted_points.append(centroid)

extracted_points = np.array(extracted_points)

# Detect duplicated cell IDs (i.e., multiple points mapping to the same cell)
duplicate_indices = []
cell_id_to_indices = {}  # Dictionary: cell_id -> list of indices in cell_ids

for i, c_id in enumerate(cell_ids):
    if c_id not in cell_id_to_indices:
        cell_id_to_indices[c_id] = [i]
    else:
        cell_id_to_indices[c_id].append(i)

# Identify and report cells with duplicates
for c_id, indices in cell_id_to_indices.items():
    if len(indices) > 1:
        print(f"Warning: Multiple input points {indices} mapped to cell {c_id}.")
        # We'll remove all but the first occurrence
        duplicate_indices.extend(indices[1:])

# If duplicates exist, remove them from cell_ids, extracted_points, and reduced_grid
if duplicate_indices:
    print("Removing repeated instances so they appear only once in final output.")
    duplicate_indices = sorted(set(duplicate_indices), reverse=True)
    
    # Convert cell_ids to a list so we can pop easily
    cell_ids_list = list(cell_ids)
    
    for idx in duplicate_indices:
        cell_ids_list.pop(idx)
        extracted_points = np.delete(extracted_points, idx, axis=0)
        reduced_grid = np.delete(reduced_grid, idx, axis=0)

    cell_ids = cell_ids_list

# Sort reduced mesh cell IDs for niceness later on
cell_ids = sorted(cell_ids)

# Export cell IDs to a file
np.savetxt(output_filename, cell_ids, delimiter=",", fmt="%d")

# Write sample points locations (reduced_grid) to a CSV file
csv_output_filename = "outputs/sample_points_locations.csv"
np.savetxt(csv_output_filename, extracted_points, delimiter=",", fmt="%.5f", header="x,y,z", comments="")
print(f"Reduced grid points have been written to {csv_output_filename}.")



# ---------- Plot the results using Matplotlib ------------
if display_reduced_mesh:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the original domain for reference
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Plot the reduced resolution grid (probed points)
    ax.scatter(reduced_grid[:, 0], reduced_grid[:, 1], color="red", label="Reduced Resolution Grid", s=10)

    # Plot the extracted points
    ax.scatter(extracted_points[:, 0], extracted_points[:, 1], color="blue", label="Mapped Centroids", s=10)

    # Draw squares for context
    square1 = patches.Rectangle((-0.5, 0.5), 1, 1, linewidth=1, edgecolor="black", facecolor="black")
    square2 = patches.Rectangle((-0.5, -1.5), 1, 1, linewidth=1, edgecolor="black", facecolor="black")
    ax.add_patch(square1)
    ax.add_patch(square2)

    # Add labels, legend, and grid
    ax.set_title("Reduced Resolution Mesh")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid()
    ax.set_aspect('equal')

    # Save the plot
    plt.savefig("outputs/reduced_mesh_vis.png", dpi=300)

    # Display plot, but do not block execution and close after 5 seconds
    plt.show(block=False)
    start_plot_timer = time.time()
    while time.time() - start_plot_timer < 3:
        plt.pause(0.1)  # Allow GUI updates during the sleep
    plt.close()

