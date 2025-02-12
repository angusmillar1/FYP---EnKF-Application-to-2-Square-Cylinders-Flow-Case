import os
import glob
import shutil
from pathlib import Path
from PIL import Image
import pyvista as pv
import time
import pandas as pd
import numpy as np

plotAll = 0
plotAvg = 1
cleanpngs = 1

start_timing = time.time()

# Parent directory
parent_dir = "outputs"

# Read points from the CSV file
points_to_add = pd.read_csv(os.path.join(parent_dir, "sample_points_locations.csv"), skiprows=1, header=None).values
points_to_add = np.hstack((points_to_add[:, 1:3],np.ones((points_to_add.shape[0], 1))))

# Set path to file directory
input_dir = os.path.join(parent_dir, "visualisations")
output_dir = os.path.join(input_dir, "animations")
gif_dir = os.path.join(output_dir, "gifs")

# Clean the output directories
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

if os.path.exists(gif_dir):
    shutil.rmtree(gif_dir)
os.makedirs(gif_dir)

# Function to add points to the PyVista plotter
def add_points(plotter, points):
    for point in points:
        plotter.add_mesh(pv.Sphere(center=point, radius=0.1), color="black")

# File patterns
patterns = [
    os.path.join(input_dir, "member*_*.vtk"),
    os.path.join(input_dir, "refSoln_*.vtk"),
]

# Get list of files
file_lists = []
for pattern in patterns:
    files = sorted(glob.glob(pattern), key=os.path.basename)
    file_lists.append(files)

# Combine file lists for processing
file_data = []
for files in file_lists:
    for file in files:
        group = Path(file).stem.split("_")[0]
        timestep = Path(file).stem.split("_")[-1]
        file_data.append((group, timestep, file))

# Sort file data by group and timestep
file_data = sorted(file_data, key=lambda x: (x[0], int(x[1])))

# Group files by animation group
grouped_files = {}
for group, timestep, file in file_data:
    if group not in grouped_files:
        grouped_files[group] = []
    grouped_files[group].append(file)

# Function to generate GIF
def generate_gif(image_files, output_gif):
    frames = [Image.open(img) for img in image_files]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # Frame duration in ms
        loop=0         # Infinite loop
    )
    print(f"Generated GIF: {output_gif}")

if plotAll:
    for group, files in grouped_files.items():
        print(f"Processing group: {group}")

        image_files = []
        for file in files:
            # Load the VTK file
            vtk_object = pv.read(file)

            # Retain only the necessary fields: "U" and "p"
            # vtk_object = vtk_object.extract_points(vtk_object.point_data.keys())
            fields_to_keep = ["U"]

            # Remove unwanted fields
            for field in list(vtk_object.point_data.keys()):  # Convert to a list to modify during iteration
                if field not in fields_to_keep:
                    vtk_object.point_data.remove(field)
            
            # Extract the X-component of U
            vtk_object["Ux"] = vtk_object["U"][:, 0]

            # Create a PyVista plotter
            plotter = pv.Plotter(off_screen=True)  # False if visualising during runtime
            plotter.window_size = [1000, 500]  # Example for 16:9 aspect ratio
            plotter.add_mesh(vtk_object, scalars="Ux", cmap="coolwarm", clim=(-1.1,2.1))
            plotter.camera_position = [
                (0, 0, 100),  # Position
                (10, 0, 0),   # Focal point
                (0, 1, 0),    # View up
            ]
            plotter.set_background("white")
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 7

            # Add points to the plot
            add_points(plotter, points_to_add)
            # plotter.show()

            # Save screenshot
            image_file = os.path.join(output_dir, f"{Path(file).stem}.png")
            plotter.screenshot(image_file)
            image_files.append(image_file)
            print(image_file)



            # Clean up plotter
            plotter.close()

        # Generate GIF for the group
        gif_file = os.path.join(gif_dir, f"{group}_animation.gif")
        generate_gif(image_files, gif_file)






# Plot only ensemble average and reference fields
if plotAvg:
    # ---------------------------
    # Generate average-field visualisations
    # ---------------------------
    # Build a dictionary mapping timestep -> list of member files (exclude refSoln)
    member_files_by_timestep = {}
    for group, timestep, file in file_data:
        # Exclude files from refSoln group (case-insensitive)
        if group.lower() != "refsoln":
            if timestep not in member_files_by_timestep:
                member_files_by_timestep[timestep] = []
            member_files_by_timestep[timestep].append(file)
    
    # Sort timesteps in increasing order (assuming they are numeric strings)
    timesteps_sorted = sorted(member_files_by_timestep.keys(), key=lambda x: int(x))
    
    image_files_avg = []
    for t in timesteps_sorted:
        # if int(t) > 800: break  # for if only a limited time range is desired
        files_t = member_files_by_timestep[t]
        sum_array = None
        count = 0
        # For each ensemble member file at this timestep, read and accumulate Ux data
        for file in files_t:
            vtk_object = pv.read(file)
            # Retain only the necessary field "U" (remove others)
            fields_to_keep = ["U"]
            for field in list(vtk_object.point_data.keys()):
                if field not in fields_to_keep:
                    vtk_object.point_data.remove(field)
            # Compute Ux from U (assumes U is a 2D array with at least one column)
            vtk_object["Ux"] = vtk_object["U"][:, 0]
            
            # Initialize or accumulate the Ux field
            if sum_array is None:
                sum_array = np.array(vtk_object["Ux"])
            else:
                sum_array += vtk_object["Ux"]
            count += 1
        
        # Calculate the average field
        avg_field = sum_array / count
        
        # Use one of the VTK objects as a template for plotting (here, we reuse the last one read)
        vtk_object["Ux"] = avg_field  # Replace Ux with the average field
        
        # Create a PyVista plotter and plot the averaged field
        plotter = pv.Plotter(off_screen=True)
        plotter.window_size = [1000, 500]
        plotter.add_mesh(vtk_object, scalars="Ux", cmap="coolwarm", clim=(-1.1,2.1))
        plotter.camera_position = [
            (0, 0, 100),  # Position
            (10, 0, 0),   # Focal point
            (0, 1, 0),    # View up
        ]
        plotter.set_background("white")
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 7
        
        # Add the circles (points) as before
        add_points(plotter, points_to_add)
        
        # Save the screenshot for this timestep (filename: avg_{t}.png)
        image_file = os.path.join(output_dir, f"avg_{t}.png")
        plotter.screenshot(image_file)
        image_files_avg.append(image_file)
        print(image_file)
        plotter.close()
    
    # Generate GIF for the averaged field
    gif_file_avg = os.path.join(gif_dir, "avg_animation.gif")
    generate_gif(image_files_avg, gif_file_avg)
    
    
    # ---------------------------
    # Generate reference solution (refSoln) visualisations
    # ---------------------------
    image_files_ref = []
    if "refSoln" in grouped_files:
        for file in grouped_files["refSoln"]:
            vtk_object = pv.read(file)
            # Retain only the necessary field "U"
            fields_to_keep = ["U"]
            for field in list(vtk_object.point_data.keys()):
                if field not in fields_to_keep:
                    vtk_object.point_data.remove(field)
            vtk_object["Ux"] = vtk_object["U"][:, 0]
    
            # Create a PyVista plotter as before
            plotter = pv.Plotter(off_screen=True)
            plotter.window_size = [1000, 500]
            plotter.add_mesh(vtk_object, scalars="Ux", cmap="coolwarm", clim=(-1.1,2.1))
            plotter.camera_position = [
                (0, 0, 100),
                (10, 0, 0),
                (0, 1, 0),
            ]
            plotter.set_background("white")
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 7
            add_points(plotter, points_to_add)
    
            # Extract timestep from filename (assumes filename structure: refSoln_{timestep}.vtk)
            timestep = Path(file).stem.split("_")[-1]
            image_file = os.path.join(output_dir, f"refSoln_{timestep}.png")
            plotter.screenshot(image_file)
            image_files_ref.append(image_file)
            print(image_file)
            plotter.close()
    
        # Generate GIF for the refSoln
        gif_file_ref = os.path.join(gif_dir, "refSoln_animation.gif")
        generate_gif(image_files_ref, gif_file_ref)


# Delete all .png files to save storage
if cleanpngs:
    for item in os.listdir(output_dir):
        file_path = os.path.join(output_dir, item)
        if os.path.isfile(file_path):
            os.remove(file_path)

end_timing = time.time()
print("Animation creation runtime = " + str(end_timing - start_timing))