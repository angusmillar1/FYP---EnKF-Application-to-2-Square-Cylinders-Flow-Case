import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import time as timepkg
import os
import re
import numpy as np
import sys

# CHOOSE WHAT TO PLOT
wholeFieldOn = 1
probePlotOn = 1
plotAssimInt = 1
printProgress = 1

if len(sys.argv) > 1 and sys.argv[1]:
    # Inherited inputs if calling from Allrun.py 
    num_members = int(sys.argv[1])  # Get number of members from parent script for spread plot
    assimInt = float(sys.argv[2])   # Get assimilation interval for plotting of vert lines
    # Plot everything when run in big sim
    wholeFieldOn = 1
    probePlotOn = 1
    plotAssimInt = 1
    printProgress = 1
else:
    # Equivalent inherited inputs if running independtly
    num_members = 15    # Manually set number of members for spread plot
    assimInt = 10       # Manually set assimilation interval for plotting of vert lines

# Other plotting inputs
probeNum = [0,1,2,7,13] # Choose probe points to plot for

if len(sys.argv) > 1 and sys.argv[1]:
    timeWindow = []     # Automatically select whole domain when run from allrun
else:
    timeWindow = [0,42] # Manually select region in time to plot, eg could be [2, 5] or left empty for whole domain.




# File path
input_path = "outputs/"
output_path = "outputs/errorPlots/"



# ADD INDICES TO SAMPLE POINTS LOCATION FILE
probe_coords_file = input_path+"sample_points_locations.csv"
df = pd.read_csv(probe_coords_file)
if 'p' not in df.columns:
    df.insert(0, 'p', range(0, len(df)))    # Insert a new column 'p' with index starting from 1
    df.to_csv(probe_coords_file, index=False)
    if printProgress: print(f"Modified file saved as {probe_coords_file}")
else: printProgress: print(f"{probe_coords_file} already contains point indices")



# WHOLE FIELD ERRORS

if wholeFieldOn:
    if printProgress: print("Starting whole field error plots")
    # Read the CSV file
    try:
        # Load the data into a DataFrame
        data = pd.read_csv(input_path+"error_write.csv")

        # Filter the data if timeWindow is provided
        if timeWindow and len(timeWindow) == 2:
            t_min, t_max = timeWindow
            data = data[(data["T"] >= t_min) & (data["T"] <= t_max)]

        # Check if the required columns exist
        required_columns = ["T", "L1_u", "L1_v", "L1_tot", "L2_u", "L2_v", "L2_tot", "MSE_u", "MSE_v", "MSE_tot"]

        if all(col in data.columns for col in required_columns):
            # ---------------------------
            # L1 Norm Plot
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(data["T"], data["L1_u"], label="L1_u", marker='o')
            plt.plot(data["T"], data["L1_v"], label="L1_v", marker='s')
            # If you wish to include pressure later, uncomment the following:
            # plt.plot(data["T"], data["L1_p"], label="L1_p", marker='^')
            plt.plot(data["T"], data["L1_tot"], label="L1_tot", marker='d')

            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel("Time (T)", fontsize=12)
            plt.ylabel("L1 Norm", fontsize=12)
            plt.title("L1 Norm Over Time", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True)
            plt.tight_layout()

            # Save the L1 plot
            if not timeWindow: output_plot_path_L1 = output_path+"L1_plot.png"
            else: output_plot_path_L1 = output_path+f"L1_plot_windowed_{t_min}_{t_max}.png"
            plt.savefig(output_plot_path_L1, dpi=300)
            if printProgress: print(f"L1 plot saved as '{output_plot_path_L1}'")

            # Display the L1 plot for 5 seconds
            plt.show(block=False)
            start_plot_timer = timepkg.time()
            while timepkg.time() - start_plot_timer < 2:
                plt.pause(0.1)
            plt.close()

            # ---------------------------
            # L2 Norm Plot
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(data["T"], data["L2_u"], label="L2_u", marker='o')
            plt.plot(data["T"], data["L2_v"], label="L2_v", marker='s')
            # If you wish to include pressure later, uncomment the following:
            # plt.plot(data["T"], data["L2_p"], label="L2_p", marker='^')
            plt.plot(data["T"], data["L2_tot"], label="L2_tot", marker='d')

            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel("Time (T)", fontsize=12)
            plt.ylabel("L2 Norm", fontsize=12)
            plt.title("L2 Norm Over Time", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True)
            plt.tight_layout()

            # Save the L2 plot
            if not timeWindow: output_plot_path_L2 = output_path+"L2_plot.png"
            else: output_plot_path_L2 = output_path+f"L2_plot_windowed_{t_min}_{t_max}.png"
            plt.savefig(output_plot_path_L2, dpi=300)
            if printProgress: print(f"L2 plot saved as '{output_plot_path_L2}'")

            # Display the L2 plot for 5 seconds
            plt.show(block=False)
            start_plot_timer = timepkg.time()
            while timepkg.time() - start_plot_timer < 2:
                plt.pause(0.1)
            plt.close()

            # ---------------------------
            # MSE Plot
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(data["T"], data["MSE_u"], label="MSE_u", marker='o')
            plt.plot(data["T"], data["MSE_v"], label="MSE_v", marker='s')
            plt.plot(data["T"], data["MSE_tot"], label="MSE_tot", marker='d')

            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel("Time (T)", fontsize=12)
            plt.ylabel("MSE", fontsize=12)
            plt.title("MSE Over Time", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True)
            plt.tight_layout()

            # Save the MSE plot
            if not timeWindow: output_plot_path_MSE = output_path+"MSE_plot.png"
            else: output_plot_path_MSE = output_path+f"MSE_plot_windowed_{t_min}_{t_max}.png"
            plt.savefig(output_plot_path_MSE, dpi=300)
            if printProgress: print(f"MSE plot saved as '{output_plot_path_MSE}'")

            # Display the MSE plot for 5 seconds
            plt.show(block=False)
            start_plot_timer = timepkg.time()
            while timepkg.time() - start_plot_timer < 2:
                plt.pause(0.1)
            plt.close()

        else:
            print(f"The file does not contain the required columns: {required_columns}")
    except FileNotFoundError:
        print(f"File '{input_path}error_write.csv' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")










# MEASUREMENT POINTS ERROR TRACKING

if probePlotOn:
    if printProgress: print("Starting probe plots")

    # Base directory where the member folders are stored.
    base_dir = "memberRunFiles"

    # Initialize a dictionary to hold the data for each member and the reference solution.
    data = {}

    # Function to extract data from a given directory structure
    def extract_data(member_dir):
        """Extracts velocity data from a given member directory."""
        time_dirs = sorted(os.listdir(member_dir), key=lambda x: float(x))
        
        times_list = []
        u_data_list = []
        v_data_list = []
        
        for t in time_dirs:
            file_path = os.path.join(member_dir, t, "U")
            if not os.path.isfile(file_path):
                continue
            
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            data_lines = lines[11:]
            
            for line in data_lines:
                if not line.strip():
                    continue

                tokens = line.strip().split()
                try:
                    time_val = float(tokens[0])
                except ValueError:
                    continue
                times_list.append(time_val)
                
                probe_strings = re.findall(r'\((.*?)\)', line)
                u_row = []
                v_row = []
                for ps in probe_strings:
                    numbers = ps.split()
                    if len(numbers) < 2:
                        continue
                    try:
                        u_val = float(numbers[0])
                        v_val = float(numbers[1])
                    except ValueError:
                        continue
                    u_row.append(u_val)
                    v_row.append(v_val)
                
                u_data_list.append(u_row)
                v_data_list.append(v_row)
        
        if times_list:
            return {
                "time": np.array(times_list),
                "u": np.array(u_data_list),
                "v": np.array(v_data_list),
            }
        else:
            return {
                "time": np.array([]),
                "u": np.array([]),
                "v": np.array([]),
            }

    # Extract ensemble member data
    for member in range(1, num_members + 1):
        member_dir = os.path.join(base_dir, f"member{member}", "postProcessing", "velocityProbes")
        data[f"member{member}"] = extract_data(member_dir)
        if printProgress: print(f"Data read for member {member}/{num_members}")

    # Extract reference solution data
    ref_dir = os.path.join(base_dir, "refSoln", "postProcessing", "velocityProbes")
    data["refSoln"] = extract_data(ref_dir)
    if printProgress: print("Data read for reference solution")

    # Sort member keys
    members = sorted([k for k in data.keys() if "member" in k], key=lambda m: int(m.replace("member", "")))
    num_members = len(members)

    # Use the time array from the first member (assuming identical simulation times).
    time = data[members[0]]["time"]
    num_timesteps = len(time)
    num_probe_points = data[members[0]]["u"].shape[1]

    # Stack the data from each member into 3D arrays.
    u_ensemble = np.empty((num_members, num_timesteps, num_probe_points))
    v_ensemble = np.empty((num_members, num_timesteps, num_probe_points))

    for i, mem in enumerate(members):
        u_ensemble[i, :, :] = data[mem]["u"]
        v_ensemble[i, :, :] = data[mem]["v"]

    # Now, if timeWindow is set, filter the ensemble data and time arrays.
    if timeWindow and len(timeWindow) == 2:
        t_min, t_max = timeWindow
        mask = (time >= t_min) & (time <= t_max)

        # Filter ensemble times and velocities
        time = time[mask]
        u_ensemble = u_ensemble[:, mask, :]
        v_ensemble = v_ensemble[:, mask, :]

    # Compute the ensemble mean, max, and min across members (axis 0).
    u_mean = np.mean(u_ensemble, axis=0)
    u_max = np.max(u_ensemble, axis=0)
    u_min = np.min(u_ensemble, axis=0)

    v_mean = np.mean(v_ensemble, axis=0)
    v_max = np.max(v_ensemble, axis=0)
    v_min = np.min(v_ensemble, axis=0)

    if printProgress: print("Ensemble statistics found")

    # Extract the reference solution data
    time_ref = data["refSoln"]["time"]
    u_ref = data["refSoln"]["u"]
    v_ref = data["refSoln"]["v"]

    # If you want to *also* filter reference solution data by the same timeWindow:
    if timeWindow and len(timeWindow) == 2 and len(time_ref) > 0:
        ref_mask = (time_ref >= t_min) & (time_ref <= t_max)
        time_ref = time_ref[ref_mask]
        u_ref = u_ref[ref_mask, :]
        v_ref = v_ref[ref_mask, :]

    # Get the positions of the probe points for clearer analysis
    probe_coords = pd.read_csv(probe_coords_file, skiprows=1, header=None).values

    # --- Plotting the results ---
    if printProgress: print("Starting plotting")

    for p in probeNum:

        # Select the relevant coordinates
        index, x_p, y_p, z_p = probe_coords[p]

        # Plot for the u component.
        plt.figure(figsize=(12, 6))
        plt.fill_between(time, u_min[:, p], u_max[:, p], alpha=0.3, color='C0', label='Ensemble Spread')
        plt.plot(time, u_mean[:, p], label='Ensemble Mean', color='C0')
        if u_ref.shape[0] > 0:
            plt.plot(time_ref, u_ref[:, p], '--', color='C2', label='Reference Solution')

        plt.xlabel("Time")
        plt.ylabel("u velocity")
        plt.title(f"Ensemble u Velocity at Probe ({x_p:.2f}, {y_p:.2f})")
        plt.grid(True)
        plt.legend(loc="best")

        if plotAssimInt:
            # Use MultipleLocator for vertical lines every 0.1 time units
            ax = plt.gca()  # Get current axes
            ax.xaxis.set_major_locator(ticker.MultipleLocator(assimInt))
            ax.grid(which='major', axis='x', linestyle='--', color='black')    

        plt.tight_layout()
        if not timeWindow: plt.savefig(output_path+f"U_probe_series_point{p}.png", dpi=300)
        else: plt.savefig(output_path+f"U_probe_series_point{p}_windowed_{t_min}_{t_max}.png", dpi=300)

        # Show for a short period
        plt.show(block=False)
        start_plot_timer = timepkg.time()
        while timepkg.time() - start_plot_timer < 2:
            plt.pause(0.1)
        plt.close()

        # Plot for the v component.
        plt.figure(figsize=(12, 6))
        plt.fill_between(time, v_min[:, p], v_max[:, p], alpha=0.3, color='C1', label='Ensemble Spread')
        plt.plot(time, v_mean[:, p], label='Ensemble Mean', color='C1')
        if v_ref.shape[0] > 0:
            plt.plot(time_ref, v_ref[:, p], '--', color='C3', label='Reference Solution')

        plt.xlabel("Time")
        plt.ylabel("v velocity")
        plt.title(f"Ensemble v Velocity at Probe ({x_p:.2f}, {y_p:.2f})")
        plt.grid(True)
        plt.legend(loc="best")

        if plotAssimInt:
            # Use MultipleLocator for vertical lines every 0.1 time units
            ax = plt.gca()  # Get current axes
            ax.xaxis.set_major_locator(ticker.MultipleLocator(assimInt))
            ax.grid(which='major', axis='x', linestyle='--', color='black')    

        plt.tight_layout()
        if not timeWindow: plt.savefig(output_path+f"V_probe_series_point{p}.png", dpi=300)
        else: plt.savefig(output_path+f"V_probe_series_point{p}_windowed_{t_min}_{t_max}.png", dpi=300)

        # Show for a short period
        plt.show(block=False)
        start_plot_timer = timepkg.time()
        while timepkg.time() - start_plot_timer < 2:
            plt.pause(0.1)
        plt.close()
