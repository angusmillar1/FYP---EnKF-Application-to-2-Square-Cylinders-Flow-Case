import subprocess
import os

# Path to the parent directory containing member directories
parent_dir = "memberRunFiles"

# Define the OpenFOAM solver and the VTK conversion command
solver_command = "source /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && SC_pimpleFoam"
vtk_command = "source /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && foamToVTK"

# Loop through each member directory
for member_dir in sorted(os.listdir(parent_dir)):
    full_path = os.path.join(parent_dir, member_dir)
    
    # Check if the path is a directory
    if os.path.isdir(full_path):
        print(f"Processing case: {member_dir}")
        
        try:
            # Run the OpenFOAM solver
            print(f"Running solver: {solver_command.rsplit(' ', 1)[-1]} in {full_path}")
            # subprocess.run(solver_command, cwd=full_path, shell=True, capture_output=True, text=True, check=True, executable="/bin/bash")  # Run without terminal output
            subprocess.run(solver_command, cwd=full_path, shell=True, check=True, executable="/bin/bash")  # Run with terminal output
            
            # Run foamToVTK after the solver finishes
            print(f"Converting results to VTK: {vtk_command.rsplit(' ', 1)[-1]} in {full_path}")
            subprocess.run(vtk_command, cwd=full_path, shell=True, capture_output=True, text=True, check=True, executable="/bin/bash")
        
        except subprocess.CalledProcessError as e:
            print(f"Error in case {member_dir}: {e}")
            continue  # Move to the next directory if an error occurs

print("All cases processed.")
