import os
import glob
import shutil
import csv
from pathlib import Path
from PIL import Image
import pyvista as pv
import time

start_timing = time.time()

# Parent directory
parent_dir = "../outputs"

# Read points from the CSV file
points_to_add = []
with open(os.path.join(parent_dir, "sample_points_locations.csv"), mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        point = [float(coord) for coord in row]
        point[2] = 1.0  # Set the z-component to 1.0
        points_to_add.append(point)

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
    os.path.join(input_dir, "Square_Cylinders_Non_Linear_Mesh1Dvlpd_*.vtk"),
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

for group, files in grouped_files.items():
    print(f"Processing group: {group}")

    image_files = []
    for file in files:
        # Load the VTK file
        vtk_object = pv.read(file)

        # Retain only the necessary fields: "U" and "p"
        # vtk_object = vtk_object.extract_points(vtk_object.point_data.keys())
        fields_to_keep = ["U", "p"]

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



        # Clean up plotter
        plotter.close()

    # Generate GIF for the group
    gif_file = os.path.join(gif_dir, f"{group}_animation.gif")
    generate_gif(image_files, gif_file)


end_timing = time.time()
print("Animation creation runtime = " + str(end_timing - start_timing))