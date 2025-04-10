import os
import glob
import shutil
from pathlib import Path
from PIL import Image
import pyvista as pv
import time
import pandas as pd
import numpy as np
import sys

if len(sys.argv) > 1 and sys.argv[1]:
    assimInt = float(sys.argv[1])       # Automatically inherit values if run from ALLRUN
    totalRuntime = int(sys.argv[2])
    writeFreq = int(sys.argv[3])
else:
    assimInt = 5                        # Manually set values if run internally
    totalRuntime = 200
    writeFreq = 100

plotAll = 0
plotAvg = 1
plotVarT = 0       # New flag for variance visualisations
plotRMS = 1
makePNGSandGIFS = 0
cleanpngs = 0
cleanvtks = 0

start_timing = time.time()

# Parent directory
parent_dir = "outputs"

# Read points from the CSV file
points_to_add = pd.read_csv(os.path.join(parent_dir, "sample_points_locations.csv"), skiprows=1, header=None).values
points_to_add = np.hstack((points_to_add[:, 1:3], np.ones((points_to_add.shape[0], 1))))

# Set path to file directory
input_dir = os.path.join(parent_dir, "visualisations")
output_dir = os.path.join(input_dir, "animations")
gif_dir = os.path.join(output_dir, "gifs")
vtk_dir = os.path.join(input_dir, "vtk")

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
    os.path.join(vtk_dir, "member*_*.vtk"),
    os.path.join(vtk_dir, "refSoln_*.vtk"),
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
        duration=500,  # Frame duration in ms
        loop=0         # Infinite loop
    )
    print(f"Generated GIF: {output_gif}")

def compute_uv_fluctuations(file_pattern, u_field, v_field, output_dir, output_filename, time_interval, remove_original=True):
    """
    Process a set of assimilation VTK files (named like "ensavg_{t}.vtk") that have timesteps which are
    nonzero multiples of time_interval. For each such snapshot, compute:
    
      - The instantaneous fluctuation fields (u_prime and v_prime)
      - The instantaneous moments: (u_prime)**2, (u_prime)*(v_prime), and (v_prime)**2 
      - The global (time-averaged over all selected snapshots) moments: 
          <(u - mean_u)**2>, <(u - mean_u)*(v - mean_v)>, and <(v - mean_v)**2>
    
    These are then stored in the VTK file with the following 8 fields:
      1. "u_prime"   : instantaneous u fluctuation
      2. "v_prime"   : instantaneous v fluctuation
      3. "inst_u2"   : instantaneous u′²
      4. "inst_uv"   : instantaneous u′v′
      5. "inst_v2"   : instantaneous v′²
      6. "global_u2" : global (time-averaged) u′²
      7. "global_uv" : global (time-averaged) u′v′
      8. "global_v2" : global (time-averaged) v′²
    
    Parameters:
      file_pattern (str)  : Glob pattern for input VTK files (e.g. "path/to/ensavg_*.vtk")
      u_field (str)       : Name of the u component field in the files (e.g. "Ux")
      v_field (str)       : Name of the v component field in the files (e.g. "Uy")
      output_dir (str)    : Directory to save the output VTK files.
      time_interval (int) : Process only files whose timestep (extracted from filename) is a nonzero multiple of this.
      remove_original (bool) : If True, remove the original u_field and v_field from point and cell data.
    """
    
    # 1. Gather and filter files:
    all_files = sorted(glob.glob(file_pattern))
    selected_files = []
    for fname in all_files:
        base = os.path.basename(fname)
        # Assumes file naming like "refSoln_500.vtk" or "ensavg_500.vtk"
        try:
            # Extract the timestep assuming it is the part after the final underscore.
            t_str = base.split('_')[-1].replace(".vtk", "")
            t_val = int(t_str)
        except ValueError:
            continue  # Skip files whose names do not conform.
        
        # Only select files with nonzero timesteps that are multiples of the specified time_interval.
        if t_val != 0 and t_val % time_interval == 0:
            selected_files.append(fname)
    
    if not selected_files:
        print("No files match the given time_interval criteria.")
        return
    
    N = len(selected_files)
    
    # 2. Pass 1: Compute global time-average fields (mean_u and mean_v).
    u_sum = None
    v_sum = None
    for fname in selected_files:
        vtk_obj = pv.read(fname)
        u_array = np.array(vtk_obj[u_field])
        v_array = np.array(vtk_obj[v_field])
        if u_sum is None:
            u_sum = u_array.copy()
            v_sum = v_array.copy()
        else:
            u_sum += u_array
            v_sum += v_array
    mean_u = u_sum / N
    mean_v = v_sum / N
    
    # 3. Pass 2: Compute global (time-averaged) second moments.
    u2_global_sum = None
    uv_global_sum = None
    v2_global_sum = None
    for fname in selected_files:
        vtk_obj = pv.read(fname)
        u_array = np.array(vtk_obj[u_field])
        v_array = np.array(vtk_obj[v_field])
        # Compute fluctuations
        u_prime = u_array - mean_u
        v_prime = v_array - mean_v
        if u2_global_sum is None:
            u2_global_sum = u_prime**2
            uv_global_sum = u_prime * v_prime
            v2_global_sum = v_prime**2
        else:
            u2_global_sum += u_prime**2
            uv_global_sum += u_prime * v_prime
            v2_global_sum += v_prime**2

    global_u2 = u2_global_sum / N  # global time-averaged u′² field
    global_uv = uv_global_sum / N  # global time-averaged u′v′ field
    global_v2 = v2_global_sum / N  # global time-averaged v′² field
    
    # 4. Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Pass 3: Process each file individually.
    for fname in selected_files:
        vtk_obj = pv.read(fname)
        u_array = np.array(vtk_obj[u_field])
        v_array = np.array(vtk_obj[v_field])
        
        # Compute instantaneous fluctuations.
        u_prime_inst = u_array - mean_u
        v_prime_inst = v_array - mean_v
        
        # Compute instantaneous second moments.
        inst_u2 = u_prime_inst**2
        inst_uv = u_prime_inst * v_prime_inst
        inst_v2 = v_prime_inst**2
        
        # Optionally remove the original fields.
        if remove_original:
            for f in [u_field, v_field]:
                if f in vtk_obj.point_data:
                    vtk_obj.point_data.remove(f)
                if f in vtk_obj.cell_data:
                    vtk_obj.cell_data.remove(f)
        
        # Add all 8 fields.
        vtk_obj["u_prime"]   = u_prime_inst     # instantaneous u fluctuations.
        vtk_obj["v_prime"]   = v_prime_inst     # instantaneous v fluctuations.
        vtk_obj["inst_u2"]   = inst_u2          # instantaneous u′².
        vtk_obj["inst_uv"]   = inst_uv          # instantaneous u′v′.
        vtk_obj["inst_v2"]   = inst_v2          # instantaneous v′².
        vtk_obj["global_u2"] = global_u2        # global (time-averaged) u′².
        vtk_obj["global_uv"] = global_uv        # global (time-averaged) u′v′.
        vtk_obj["global_v2"] = global_v2        # global (time-averaged) v′².
        
        # Extract the timestep identifier from the filename.
        base = os.path.basename(fname)
        t_str = base.replace("ensavg_", "").replace(".vtk", "")
        
        # Save the new VTK file as "ensfluct_{t}.vtk" in the output directory.
        output_file = os.path.join(output_dir, f"{output_filename}_{t_str}.vtk")
        vtk_obj.save(output_file)

if plotAll:
    for group, files in grouped_files.items():
        print(f"Processing group: {group}")

        image_files = []
        for file in files:
            # Load the VTK file
            vtk_object = pv.read(file)

            # Retain only the necessary fields: "U"
            fields_to_keep = ["U"]
            for field in list(vtk_object.point_data.keys()):
                if field not in fields_to_keep:
                    vtk_object.point_data.remove(field)
            
            # Extract the X-component of U
            vtk_object["Ux"] = vtk_object["U"][:, 0]

            # Create a PyVista plotter
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

            # Add points to the plot
            add_points(plotter, points_to_add)

            # Save screenshot
            image_file = os.path.join(output_dir, f"{Path(file).stem}.png")
            plotter.screenshot(image_file)
            image_files.append(image_file)
            print(image_file)

            plotter.close()

        # Generate GIF for the group
        gif_file = os.path.join(gif_dir, f"{group}_animation.gif")
        generate_gif(image_files, gif_file)

if plotAvg:
    # ---------------------------
    # Generate average-field visualisations
    # ---------------------------
    print("Generating ensemble average plots")
    # Build a dictionary mapping timestep -> list of member files (exclude refSoln)
    member_files_by_timestep = {}
    for group, timestep, file in file_data:
        if group.lower() != "refsoln":
            if timestep not in member_files_by_timestep:
                member_files_by_timestep[timestep] = []
            member_files_by_timestep[timestep].append(file)
    
    # Sort timesteps in increasing order
    timesteps_sorted = sorted(member_files_by_timestep.keys(), key=lambda x: int(x))
    
    image_files_avg = []
    for t in timesteps_sorted:
        print(f"{int(int(t)/writeFreq)}/{totalRuntime}")
        files_t = member_files_by_timestep[t]
        sum_array_Ux = None
        sum_array_Uy = None
        count = 0
        # For each ensemble member file at this timestep, read and accumulate Ux data
        for file in files_t:
            vtk_object = pv.read(file)
            fields_to_keep = ["U"]
            for field in list(vtk_object.point_data.keys()):
                if field not in fields_to_keep:
                    vtk_object.point_data.remove(field)
            vtk_object["Ux"] = vtk_object["U"][:, 0]
            vtk_object["Uy"] = vtk_object["U"][:, 1]
            
            if sum_array_Ux is None:
                sum_array_Ux = np.array(vtk_object["Ux"])
                sum_array_Uy = np.array(vtk_object["Uy"])
            else:
                sum_array_Ux += vtk_object["Ux"]
                sum_array_Uy += vtk_object["Uy"]
            count += 1
        
        Ux_avg_field = sum_array_Ux / count
        Uy_avg_field = sum_array_Uy / count
        
        # Use one of the VTK objects as a template for plotting
        vtk_object["Ux"] = Ux_avg_field
        vtk_object["Uy"] = Uy_avg_field

        # Remove the original fields so only Ux and Uy remain
        if "U" in vtk_object.point_data: vtk_object.point_data.remove("U")
        if "U" in vtk_object.cell_data: vtk_object.cell_data.remove("U")
        if "p" in vtk_object.cell_data: vtk_object.cell_data.remove("p")
        if "cellID" in vtk_object.cell_data: vtk_object.cell_data.remove("cellID")

        # SAVE THE AVERAGE FIELDS AT EACH TIMESTEP TO avgvtk_dir titled ensavg_{t}.vtk
        output_file = os.path.join(vtk_dir, f"ensavg_{t}.vtk")
        vtk_object.save(output_file)
        
        if not makePNGSandGIFS: continue
        
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
        
        image_file = os.path.join(output_dir, f"avg_{t}.png")
        plotter.screenshot(image_file)
        image_files_avg.append(image_file)
        # print(image_file)
        plotter.close()

    if makePNGSandGIFS:
        gif_file_avg = os.path.join(gif_dir, "avg_animation.gif")
        generate_gif(image_files_avg, gif_file_avg)
        shutil.copy(gif_file_avg, "outputs/errorPlots")
    
    # ---------------------------
    # Generate reference solution (refSoln) visualisations
    # ---------------------------
    print("Generating reference solution plots")
    image_files_ref = []
    if "refSoln" in grouped_files:
        for file in grouped_files["refSoln"]:
            vtk_object = pv.read(file)
            vtk_object["Ux"] = vtk_object["U"][:, 0]
            vtk_object["Uy"] = vtk_object["U"][:, 1]

            # Remove the original fields so only Ux and Uy remain
            if "U" in vtk_object.point_data: vtk_object.point_data.remove("U")
            if "U" in vtk_object.cell_data: vtk_object.cell_data.remove("U")
            if "p" in vtk_object.cell_data: vtk_object.cell_data.remove("p")
            if "cellID" in vtk_object.cell_data: vtk_object.cell_data.remove("cellID")

            # SAVE THE AVERAGE FIELDS AT EACH TIMESTEP TO avgvtk_dir titled ensavg_{t}.vtk
            output_file = os.path.join(vtk_dir, f"refavg_{t}.vtk")
            vtk_object.save(output_file)

            if not makePNGSandGIFS: continue
    
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
    
            timestep = Path(file).stem.split("_")[-1]
            image_file = os.path.join(output_dir, f"refSoln_{timestep}.png")
            plotter.screenshot(image_file)
            image_files_ref.append(image_file)
            # print(image_file)
            plotter.close()
    
        if makePNGSandGIFS:
            gif_file_ref = os.path.join(gif_dir, "refSoln_animation.gif")
            generate_gif(image_files_ref, gif_file_ref)
            shutil.copy(gif_file_ref, "outputs/errorPlots")


if plotVarT:
    # ---------------------------
    # Generate variance (uu, uv, vv) visualisations
    # ---------------------------
    # Build a dictionary mapping timestep -> list of member files (exclude refSoln)
    member_files_by_timestep = {}
    for group, timestep, file in file_data:
        if group.lower() != "refsoln":
            if timestep not in member_files_by_timestep:
                member_files_by_timestep[timestep] = []
            member_files_by_timestep[timestep].append(file)
    
    timesteps_sorted = sorted(member_files_by_timestep.keys(), key=lambda x: int(x))
    
    # Lists to store image filenames for animations
    image_files_uu = []
    image_files_uv = []
    image_files_vv = []
    
    # Define color limits for each variance field (adjust these as needed)
    clim_dict = {
        "uu": (0, 2.1),
        "uv": (-1.1, 1.1),
        "vv": (0, 2.1),
    }
    
    for t in timesteps_sorted:
        files_t = member_files_by_timestep[t]
        u_list = []
        v_list = []
        # Read each member file and extract u and v components
        for file in files_t:
            vtk_object = pv.read(file)
            fields_to_keep = ["U"]
            for field in list(vtk_object.point_data.keys()):
                if field not in fields_to_keep:
                    vtk_object.point_data.remove(field)
            # Assumes U has at least 2 columns: u = U[:,0], v = U[:,1]
            u_list.append(vtk_object["U"][:, 0])
            v_list.append(vtk_object["U"][:, 1])
        
        u_array = np.array(u_list)  # Shape: (n_members, n_points)
        v_array = np.array(v_list)
        
        # Compute ensemble means for u and v
        u_mean = np.mean(u_array, axis=0)
        v_mean = np.mean(v_array, axis=0)
        
        # Compute variances and covariance
        uu_field = np.mean((u_array - u_mean)**2, axis=0)
        vv_field = np.mean((v_array - v_mean)**2, axis=0)
        uv_field = np.mean((u_array - u_mean) * (v_array - v_mean), axis=0)
        
        # Use one of the VTK objects as a template (last one read)
        vtk_template = pv.read(files_t[-1])
        # It can be useful to remove extra fields if needed
        fields_to_keep = ["U"]
        for field in list(vtk_template.point_data.keys()):
            if field not in fields_to_keep:
                vtk_template.point_data.remove(field)
        
        # For each variance field, create a separate plot and save the screenshot
        for var_name, field_data, clim in zip(["uu", "uv", "vv"],
                                              [uu_field, uv_field, vv_field],
                                              [clim_dict["uu"], clim_dict["uv"], clim_dict["vv"]]):
            vtk_template[var_name] = field_data
            plotter = pv.Plotter(off_screen=True)
            plotter.window_size = [1000, 500]
            plotter.add_mesh(vtk_template, scalars=var_name, cmap="coolwarm", clim=clim)
            plotter.camera_position = [
                (0, 0, 100),
                (10, 0, 0),
                (0, 1, 0),
            ]
            plotter.set_background("white")
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 7
            add_points(plotter, points_to_add)
            
            image_file = os.path.join(output_dir, f"var_{var_name}_{t}.png")
            plotter.screenshot(image_file)
            print(image_file)
            if var_name == "uu":
                image_files_uu.append(image_file)
            elif var_name == "uv":
                image_files_uv.append(image_file)
            elif var_name == "vv":
                image_files_vv.append(image_file)
            plotter.close()
    
    # Generate GIF animations for each variance field
    gif_file_uu = os.path.join(gif_dir, "var_uu_animation.gif")
    generate_gif(image_files_uu, gif_file_uu)
    shutil.copy(gif_file_uu, "outputs/errorPlots")
    
    gif_file_uv = os.path.join(gif_dir, "var_uv_animation.gif")
    generate_gif(image_files_uv, gif_file_uv)
    shutil.copy(gif_file_uv, "outputs/errorPlots")
    
    gif_file_vv = os.path.join(gif_dir, "var_vv_animation.gif")
    generate_gif(image_files_vv, gif_file_vv)
    shutil.copy(gif_file_vv, "outputs/errorPlots")

if plotRMS:
    print("Computing ensemble r.m.s.")
    compute_uv_fluctuations(
        file_pattern=os.path.join(vtk_dir,"ensavg_*.vtk"),
        u_field="Ux",
        v_field="Uy",
        output_dir=vtk_dir,
        output_filename="ensfluct",
        time_interval=writeFreq*assimInt,
        remove_original=False
    )

    print("Computing reference r.m.s.")
    compute_uv_fluctuations(
        file_pattern=os.path.join(vtk_dir,"refavg_*.vtk"),
        u_field="Ux",
        v_field="Uy",
        output_dir=vtk_dir,
        output_filename="reffluct",
        time_interval=writeFreq*assimInt,
        remove_original=False
    )


# Delete all .png files to save storage
if cleanpngs:
    for item in os.listdir(output_dir):
        file_path = os.path.join(output_dir, item)
        if os.path.isfile(file_path):
            os.remove(file_path)

if cleanvtks:
    for item in os.listdir(vtk_dir):
        file_path = os.path.join(vtk_dir, item)
        if os.path.isfile(file_path):
            os.remove(file_path)

end_timing = time.time()
print("Animation creation runtime = " + str(end_timing - start_timing))
