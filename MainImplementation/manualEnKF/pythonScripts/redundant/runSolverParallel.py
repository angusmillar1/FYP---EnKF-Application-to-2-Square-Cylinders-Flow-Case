import subprocess
import os

# Path to the parent directory containing member directories
parent_dir = "memberRunFiles"

# Define the OpenFOAM solver and the VTK conversion command
solver_command = "bash -c '. /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && SC_pimpleFoam'"
vtk_command = "bash -c '. /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && foamToVTK'"

# Collect all member directories
member_dirs = [os.path.join(parent_dir, member_dir) for member_dir in sorted(os.listdir(parent_dir)) if os.path.isdir(os.path.join(parent_dir, member_dir))]

# Keep track of processes
processes = []

# Start running simulations for each member directory
for member_dir in member_dirs:
    print(member_dir + " running")
    solver_process = subprocess.Popen(
        solver_command, shell=True, cwd=member_dir,
        stdout=open(os.path.join(member_dir, "log.solver"), "w"),
        stderr=subprocess.STDOUT
    )
    # Add solver process to the list
    processes.append((solver_process, member_dir))

# Monitor solver processes and launch foamToVTK
vtk_processes = []  # Separate list for foamToVTK processes
for solver_process, member_dir in processes:
    solver_process.wait()  # Wait for solver to finish
    if solver_process.returncode == 0:  # Check if solver finished successfully
        print(member_dir + " processing")
        vtk_process = subprocess.Popen(
            vtk_command, shell=True, cwd=member_dir,
            stdout=open(os.path.join(member_dir, "log.foamToVTK"), "w"),
            stderr=subprocess.STDOUT
        )
        vtk_processes.append((vtk_process, member_dir))
    else:
        print(f"Solver failed for {member_dir}. Check log.solver for details.")

# Wait for all foamToVTK processes to complete
for vtk_process, member_dir in vtk_processes:
    vtk_process.wait()

print("All simulations and foamToVTK conversions are complete!")
