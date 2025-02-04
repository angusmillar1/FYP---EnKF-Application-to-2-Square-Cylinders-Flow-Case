import pandas as pd
import matplotlib.pyplot as plt
import time as timepkg
import os
import re
import numpy as np

# WHOLE FIELD ERRORS

# File path
file_path = "outputs/error_write.csv"
output_plot_path = "outputs/error_plot.png"

# Read the CSV file
try:
    # Load the data into a DataFrame
    data = pd.read_csv(file_path)

    # Check if the required columns exist
    # required_columns = ["T", "L1_u", "L1_v", "L1_p", "L1_tot", "L2_u", "L2_v", "L2_p", "L2_tot"]
    required_columns = ["T", "L1_u", "L1_v", "L1_tot", "L2_u", "L2_v", "L2_tot"]

    if all(col in data.columns for col in required_columns):
        # ---------------------------
        # L1 Norm Plot
        # ---------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(data["T"], data["L1_u"], label="L1_u", marker='o')
        plt.plot(data["T"], data["L1_v"], label="L1_v", marker='s')
        # If you wish to include pressure later, uncomment the following line:
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
        output_plot_path_L1 = "outputs/L1_plot.png"
        plt.savefig(output_plot_path_L1, dpi=300)
        print(f"L1 plot saved as '{output_plot_path_L1}'")

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
        # If you wish to include pressure later, uncomment the following line:
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
        output_plot_path_L2 = "outputs/L2_plot.png"
        plt.savefig(output_plot_path_L2, dpi=300)
        print(f"L2 plot saved as '{output_plot_path_L2}'")

        # Display the L2 plot for 5 seconds
        plt.show(block=False)
        start_plot_timer = timepkg.time()
        while timepkg.time() - start_plot_timer < 2:
            plt.pause(0.1)
        plt.close()

    else:
        print(f"The file does not contain the required columns: {required_columns}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")





# MEASUREMENT POINTS ERROR TRACKING


# Define the number of members.
num_members = 2  # Change this to the number of members you have.

# Base directory where the member folders are stored.
base_dir = "memberRunFiles"

# Initialize a dictionary to hold the data for each member.
# For each member, we will store a dictionary with keys: "time", "u", and "v".
data = {}

for member in range(1, num_members + 1):
    # Construct the path to the velocityProbes folder for this member.
    member_dir = os.path.join(base_dir, f"member{member}", "postProcessing", "velocityProbes")
    
    # Get the list of time directories under velocityProbes.
    # We assume these folder names can be converted to floats so we can sort them.
    time_dirs = sorted(os.listdir(member_dir), key=lambda x: float(x))
    
    # Lists to accumulate data across all time directories.
    times_list = []
    u_data_list = []  # Each element will be a list (one row) of u values for all probe points.
    v_data_list = []  # Each element will be a list (one row) of v values for all probe points.
    
    # Loop over each time directory.
    for t in time_dirs:
        file_path = os.path.join(member_dir, t, "U")
        if not os.path.isfile(file_path):
            continue  # Skip if the file does not exist.
        
        # Open the file and read its lines.
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Skip the first 11 lines (header lines).
        data_lines = lines[11:]
        
        # Process each remaining line.
        for line in data_lines:
            # Skip empty lines if any.
            if not line.strip():
                continue

            # The first token in the line is the time stamp.
            tokens = line.strip().split()
            try:
                time_val = float(tokens[0])
            except ValueError:
                # In case the first token isn't a valid float, skip this line.
                continue
            times_list.append(time_val)
            
            # Use a regex to extract all parenthesized groups.
            # Each group should look like "u v w".
            probe_strings = re.findall(r'\((.*?)\)', line)
            u_row = []
            v_row = []
            for ps in probe_strings:
                # Split the group string into numbers.
                numbers = ps.split()
                if len(numbers) < 2:
                    continue  # Skip if not enough numbers.
                try:
                    u_val = float(numbers[0])
                    v_val = float(numbers[1])
                except ValueError:
                    continue
                u_row.append(u_val)
                v_row.append(v_val)
            
            # Append the row for this timestep.
            u_data_list.append(u_row)
            v_data_list.append(v_row)
    
    # Convert the collected lists into NumPy arrays.
    # Each row corresponds to one timestep and each column to one probe point.
    if times_list:
        times_array = np.array(times_list)
        u_array = np.array(u_data_list)
        v_array = np.array(v_data_list)
    else:
        times_array = np.array([])
        u_array = np.array([])
        v_array = np.array([])
    
    # Store the data for this member.
    data[f"member{member}"] = {"time": times_array, "u": u_array, "v": v_array}

# At this point, the variable 'data' is a dictionary with keys "member1", "member2", ...
# Each entry is a dictionary with:
#   "time": a 1D NumPy array of time stamps,
#   "u": a 2D NumPy array of u velocity components (rows: timesteps, columns: probe points),
#   "v": a 2D NumPy array of v velocity components.

# For example, to access member 1's u data:
# print("Member 1 u data shape:", data["member1"]["u"].shape)
# print("Member 1 time data shape:", data["member1"]["time"].shape)

# Sort member keys (e.g., 'member1', 'member2', ...) to ensure a consistent order.
members = sorted(data.keys(), key=lambda m: int(m.replace("member", "")))
num_members = len(members)

# We assume that the simulation times are identical for every member. 
# Use the time array from the first member.
time = data[members[0]]["time"]
num_timesteps = len(time)
num_probe_points = data[members[0]]["u"].shape[1]

# Stack the data from each member into 3D arrays.
# u_ensemble will have shape (num_members, num_timesteps, num_probe_points)
u_ensemble = np.empty((num_members, num_timesteps, num_probe_points))
v_ensemble = np.empty((num_members, num_timesteps, num_probe_points))

for i, mem in enumerate(members):
    u_ensemble[i, :, :] = data[mem]["u"]
    v_ensemble[i, :, :] = data[mem]["v"]

# Compute the ensemble mean, max, and min across members (axis 0).
u_mean = np.mean(u_ensemble, axis=0)
u_max = np.max(u_ensemble, axis=0)
u_min = np.min(u_ensemble, axis=0)

v_mean = np.mean(v_ensemble, axis=0)
v_max = np.max(v_ensemble, axis=0)
v_min = np.min(v_ensemble, axis=0)

# --- Plotting the results ---

# Choose probe point to plot for
p = 5

# Plot for the u component.
plt.figure(figsize=(12, 6))
# Loop over each probe point (each column in our 2D arrays)
# Shade the region between the min and max values for this probe point.
plt.fill_between(time, u_min[:, p], u_max[:, p], alpha=0.3, color='C0')
# Plot the ensemble mean as a line.
plt.plot(time, u_mean[:, p], label=f'Probe {p} mean' if p==0 else None, color='C0')
plt.xlabel("Time")
plt.ylabel("u velocity")
plt.title("Ensemble u Velocity: Mean and Spread")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("outputs/U_probe_series.png", dpi=300)
plt.show(block=False)
start_plot_timer = timepkg.time()
while timepkg.time() - start_plot_timer < 2:
    plt.pause(0.1)
plt.close()

# Plot for the v component.
plt.figure(figsize=(12, 6))
plt.fill_between(time, v_min[:, p], v_max[:, p], alpha=0.3, color='C1')
plt.plot(time, v_mean[:, p], label=f'Probe {p} mean' if p==0 else None, color='C1')
plt.xlabel("Time")
plt.ylabel("v velocity")
plt.title("Ensemble v Velocity: Mean and Spread")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("outputs/V_probe_series.png", dpi=300)
plt.show(block=False)
start_plot_timer = timepkg.time()
while timepkg.time() - start_plot_timer < 2:
    plt.pause(0.1)
plt.close()

