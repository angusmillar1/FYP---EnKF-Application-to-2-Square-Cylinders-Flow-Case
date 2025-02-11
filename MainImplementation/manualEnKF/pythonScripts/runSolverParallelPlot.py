import warnings
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread will likely fail.")


import os
import re
import subprocess
import matplotlib.pyplot as plt
import threading
import time
import sys
# import warnings
# warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")

start_sim_timing = time.time()  # Time runtime 

# Target runtime for all cases
target_runtime = float(sys.argv[1])
start_time = float(sys.argv[2])
prog_endtime = float(sys.argv[3])

# Path to the parent directory containing member directories
parent_dir = "memberRunFiles"

# Define the OpenFOAM solver and VTK conversion commands
solver_command = "bash -c '. /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && SC_pimpleFoam'"
vtk_command = f"bash -c '. /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && foamToVTK -time {start_time}:{start_time+target_runtime}'"  # Convert all written files (only at current time step to reduce cost)
# vtk_command = f"bash -c '. /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && foamToVTK -time {target_runtime}'"  # Only at end time

# Collect all member directories
member_dirs = [os.path.join(parent_dir, member_dir) for member_dir in sorted(os.listdir(parent_dir)) if os.path.isdir(os.path.join(parent_dir, member_dir))]

# To store timestep and clock time data for plotting
case_time_data = {member_dir: 0.0 for member_dir in member_dirs}  # Store current simulation time for each case
member1_clock_time = 0.0  # To track ClockTime from member1 logs



# Function to monitor log files and update plotting data
def monitor_logs():
    global member1_clock_time
    while True:
        for member_dir in member_dirs:
            log_file = os.path.join(member_dir, "log.solver")
            if os.path.exists(log_file):
                with open(log_file, "r") as log:
                    for line in log:
                        # Match "Time = X"
                        time_match = re.search(r"^Time = ([\d\.]+)", line)
                        if time_match:
                            current_time = float(time_match.group(1))
                            # Update the latest simulation time for this case
                            case_time_data[member_dir] = current_time
                        
                        # Match "ClockTime = X s" and update member1_clock_time for member1
                        if member_dir == member_dirs[0]:  # Track only for member1
                            clock_time_match = re.search(r"ClockTime = ([\d\.]+) s", line)
                            if clock_time_match:
                                member1_clock_time = float(clock_time_match.group(1))
        time.sleep(2)  # Check for updates every 2 seconds

# Function to plot data in real-time as a bar chart
def live_plot():
    global member1_clock_time
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    manager = plt.get_current_fig_manager() # Get the current figure manager
    manager.window.geometry("1000x600+0+0")  # Width x Height + X Offset + Y Offset

    bar_labels = []
    for member in member_dirs:
        basename = os.path.basename(member)
        if basename.startswith("member"):
            label = "mem" + basename[6:]
        else:
            label = basename
        bar_labels.append(label)

    while True:
        ax.clear()
        # Prepare data for the bar chart
        bar_positions = range(len(member_dirs))
        bar_heights = [case_time_data[member] for member in member_dirs]
        # bar_labels = [os.path.basename(member) for member in member_dirs]
        
        # Plot bars
        ax.bar(bar_positions, bar_heights, tick_label=bar_labels, color="skyblue", alpha=0.8)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.axhline(prog_endtime, color="green", linestyle="-", linewidth=1.5, label=f"Program Runtime = {prog_endtime:.2f}")
        ax.axhline(target_runtime, color="red", linestyle="--", linewidth=1.5, label=f"End Time = {target_runtime:.2f}")
        ax.axhline(start_time, color="black", linestyle="--", linewidth=1.5, label=f"Start Time = {start_time:.2f}")
        
        # Add timer on the plot
        ax.text(0.5, 0.9, f"ClockTime (member1): {member1_clock_time:.2f} s", 
                transform=ax.transAxes, fontsize=12, color="black", ha="center", bbox=dict(facecolor="white", alpha=0.8))
        
        # Add labels and legend
        ax.set_xlabel("Case")
        ax.set_ylabel("Simulation Time")
        ax.set_title("Simulation Progress")
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.ylim(0,prog_endtime*1.1)

        plt.pause(2)  # Update the plot every 2 seconds


# Start monitoring and plotting in parallel threads
threading.Thread(target=monitor_logs, daemon=True).start()
threading.Thread(target=live_plot, daemon=True).start()

# Run simulations
processes = []
for member_dir in member_dirs:
    # print(member_dir + " running")
    log_file = os.path.join(member_dir, "log.solver")
    solver_process = subprocess.Popen(
        solver_command, shell=True, cwd=member_dir,
        stdout=open(log_file, "w"), stderr=subprocess.STDOUT
    )
    processes.append((solver_process, member_dir, log_file))

# Monitor solver processes
for solver_process, member_dir, log_file in processes:
    solver_process.wait()
    if solver_process.returncode == 0:
        # print(f"Solver completed for {member_dir}. Starting foamToVTK...")
        vtk_process = subprocess.Popen(
            vtk_command, shell=True, cwd=member_dir,
            stdout=open(os.path.join(member_dir, "log.foamToVTK"), "w"),
            stderr=subprocess.STDOUT
        )
        vtk_process.wait()
    else:
        print(f"Error: Solver failed for {member_dir}. Check {log_file} for details.")

print("All simulations and foamToVTK conversions are complete")

# Time runtime 
end_sim_timing = time.time()
print(f"Simulation elapsed time: {end_sim_timing - start_sim_timing:.2f} seconds")
