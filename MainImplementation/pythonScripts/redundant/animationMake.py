import os
from paraview.simple import *
import glob
from pathlib import Path
from PIL import Image
import sys
import csv
import time

start_timing = time.time()

# # Define the points to be added (x, y, z)
# points_to_add = [
#     [2.0, 1.0, 1.0],   
#     [4.0, 1.0, 1.0],  
#     [6.0, 1.0, 1.0],
#     [8.0, 1.0, 1.0],
#     [10.0, 1.0, 1.0],
#     [2.0, -1.0, 1.0],   
#     [4.0, -1.0, 1.0],  
#     [6.0, -1.0, 1.0],
#     [8.0, -1.0, 1.0],
#     [10.0, -1.0, 1.0]
# ]

# Read points from the CSV file
points_to_add = []
with open("../outputs/sample_points_locations.csv", mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        point = [float(coord) for coord in row]  # Convert strings to floats
        point[2] = 1.0  # Set the z-component to 1.0
        points_to_add.append(point)


# Set path to file directory
input_dir = "../outputs/visualisations/"

# Define output directory
output_dir = input_dir + "animations"
gif_dir = os.path.join(output_dir, "gifs")

# Clean the output directory
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    os.makedirs(output_dir)  # Create the directory if it doesn't exist

# Clean the output gif directory
if os.path.exists(gif_dir):
    for file in os.listdir(gif_dir):
        file_path = os.path.join(gif_dir, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    os.makedirs(gif_dir)  # Create the directory if it doesn't exist


# Function to add points as sources
def add_points(points):
    for point in points:
        sphere = Sphere()
        sphere.Radius = 0.1  # Adjust point size
        sphere.Center = point
        sphereDisplay = Show(sphere)
        sphereDisplay.DiffuseColor = [0, 0, 0]  # Black circles

# File patterns
patterns = [
    input_dir + "member*_*.vtk",
    input_dir + "Square_Cylinders_Non_Linear_Mesh1Rand_*.vtk",
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
        # Determine group name for GIF (e.g., "member1")
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
        vtk_object = OpenDataFile(file)

        # Retain only necessary fields
        pass_arrays = PassArrays(Input=vtk_object)
        pass_arrays.PointDataArrays = ["U", "p"]  # Keep only the U and p fields

        # Use filtered data in Calculator
        calculator = Calculator(Input=pass_arrays)
        calculator.ResultArrayName = "Ux"
        calculator.Function = "U_X"

        # Show the mesh
        RenderView1 = GetActiveViewOrCreate('RenderView')
        SetViewProperties(Background=[1.0, 1.0, 1.0], UseColorPaletteForBackground=0)
        RenderView1.CameraPosition = [0, 0, 100]
        RenderView1.CameraFocalPoint = [10, 0, 0]
        RenderView1.CameraParallelProjection = 1
        RenderView1.CameraParallelScale = 6  # Adjust zoom (smaller => more zoomed in)
        RenderView1.EnableRayTracing = 0
        RenderView1.ViewSize = [1000, 500]  # Width x Height
        display = Show(calculator, RenderView1)
        display.Representation = 'Surface'
        display.ColorArrayName = ["POINTS", "Ux"]
        display.LookupTable = GetColorTransferFunction("Ux")

        # Customize the color map
        color_map = GetColorTransferFunction("Ux")
        color_map.ApplyPreset("Cool to Warm", True)
        color_map.RescaleTransferFunction(-1.1, 2.1)  # Adjust range as needed

        # Add points
        add_points(points_to_add)

        

        # Turn off axes
        RenderView1.OrientationAxesVisibility = 0
        RenderView1.CenterAxesVisibility = 0

        # Force render to apply settings
        Render()

        # Save screenshot
        image_file = os.path.join(output_dir, f"{Path(file).stem}.png")
        SaveScreenshot(image_file, RenderView1)
        image_files.append(image_file)

        # Clean up
        Delete(vtk_object)
        del vtk_object

    # Generate GIF for the group
    gif_file = os.path.join(gif_dir, f"{group}_animation.gif")
    generate_gif(image_files, gif_file)

end_timing = time.time()
print("Animation creation runtime = " + str(end_timing - start_timing))

