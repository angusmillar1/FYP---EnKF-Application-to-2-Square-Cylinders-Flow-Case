import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import re
import pandas as pd

# Specify whether or not to plot output
display_output = 0  # 1 / 0

# Inputs
# vtk_file_path = "../memberRunFiles/member1/VTK/member1_0.vtk"  # Path to the VTK file
vtk_file_path = sys.argv[1]
cell_ids_file = "outputs/reduced_mesh_cell_ids.csv"  # Path to the cell IDs file

# Check if it's the initial conditions or not
last_underscore_index = vtk_file_path.rfind('_')  # Finds the last underscore
decimal_point_index = vtk_file_path.find('.', last_underscore_index)  # Finds the first '.' after the last underscore

# Extract the number between the last underscore and decimal point
timestep = int(vtk_file_path[last_underscore_index + 1:decimal_point_index])

# Set name of destination file
match = re.search(r'/member(\d+)/', vtk_file_path)
if match:
    output_filename = f"member{match.group(1)}"  # Member file
else:
    output_filename = "refSoln"  # Exact solution

# Load the VTK mesh file
mesh = pv.read(vtk_file_path)
num_cells = mesh.n_cells

if timestep != 0:
    # Load the cell IDs
    cell_ids = np.loadtxt(cell_ids_file, delimiter=",", dtype=int)

    # Validate cell IDs
    if np.any(cell_ids >= num_cells) or np.any(cell_ids < 0):
        raise ValueError("Cell IDs in the file are out of bounds for the mesh.")

# Extract full mesh data
full_cell_centers = mesh.cell_centers().points
velocity_data = mesh.cell_data["U"]  # Velocity data (e.g., "U")
Ux_data = velocity_data[:, 0]  # X-component of velocity
Uy_data = velocity_data[:, 1]  # Y-component of velocity
# P_data = mesh.cell_data["p"]  # Pressure data
full_IDs = mesh.cell_data["cellID"]  # Cell IDs

if timestep != 0:
    # Extract reduced mesh data
    reduced_cell_centers = full_cell_centers[cell_ids]
    reduced_velocity_data = velocity_data[cell_ids]
    reduced_Ux_data = reduced_velocity_data[:, 0]
    reduced_Uy_data = reduced_velocity_data[:, 1]
    # reduced_P_data = P_data[cell_ids]

# Plotting the data
if display_output:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Full mesh plot with reduced mesh overlay
    sc1 = ax1.scatter(full_cell_centers[:, 0], full_cell_centers[:, 1], c=Ux_data, cmap="coolwarm", s=10, edgecolor='none')
    if timestep != 0: ax1.scatter(reduced_cell_centers[:, 0], reduced_cell_centers[:, 1], c="black", s=20, edgecolor='black', label="Reduced Points")
    fig.colorbar(sc1, ax=ax1, label="U_x (Full Mesh)")
    ax1.set_title("Full Mesh with Reduced Points Overlay")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal", adjustable='box')
    ax1.legend()

    if timestep != 0:
        # Reduced mesh plot only
        sc2 = ax2.scatter(reduced_cell_centers[:, 0], reduced_cell_centers[:, 1], c=reduced_Ux_data, cmap="coolwarm", s=50, edgecolor='none')
        fig.colorbar(sc2, ax=ax2, label="U_x (Reduced Mesh)")
        ax2.set_title("Reduced Mesh Data")
        ax2.set_xlabel("X") 
        ax2.set_ylabel("Y")
        ax2.set_aspect("equal", adjustable='box')

    # Draw squares for context
    for ax in [ax1, ax2]:
        square1 = patches.Rectangle((-0.5, 0.5), 1, 1, linewidth=1, edgecolor="black", facecolor="none")
        square2 = patches.Rectangle((-0.5, -1.5), 1, 1, linewidth=1, edgecolor="black", facecolor="none")
        ax.add_patch(square1)
        ax.add_patch(square2)

    # Adjust layout and show the plots
    fig.suptitle(vtk_file_path, fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
    start_time = time.time()
    while time.time() - start_time < 5:
        plt.pause(0.1)  # Allow GUI updates during the sleep
    plt.close()

# --------- Write data to .csv files in vector format ----------

if timestep != 0:
    # Reduced Data

    # Create a DataFrame
    export_data = {
        "Ux": reduced_Ux_data,
        "Uy": reduced_Uy_data,
        # "p": reduced_P_data,
        "CellID": cell_ids
    }
    df = pd.DataFrame(export_data)

    # Write to CSV
    output_file = "EnKFMeshData/reducedMeshData/" + output_filename + ".csv"
    df.to_csv(output_file, index=False)

# Full Data

# Create a DataFrame
full_export_data = {
    "Ux": Ux_data,
    "Uy": Uy_data,
    # "p": P_data,
    "CellID": full_IDs
}
df = pd.DataFrame(full_export_data)

# Write to CSV
output_file = "EnKFMeshData/fullMeshData/" + output_filename + ".csv"
df.to_csv(output_file, index=False)
