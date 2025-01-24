import os
import subprocess
import sys
import time
import matplotlib.pyplot as plt

start_whole_timing = time.time()  # Time runtime

# PREDEFINED INPUTS

# Initial parameters
mesh_num = 1  # Select the Mesh to use
init_runtime = 0.1  # Set the time for the members to initially evolve before informing
file_write_freq = 10  # Frequency at which to write out data, assuming deltaT=0.01 (100=>T=1)
IC_type = "dev"  # "rand" / "dev". Define initial conditiion to use, either random of developed solution
exact_soln_path = "../referenceSolutions/Mesh1DevT1000/Square_Cylinders_Non_Linear_Mesh1DvlpdTs10_"  # Make sure this matches IC and mesh type choice

# Ensemble and filtering parameters
num_members = 8  # Set the number of ensemble members
num_loops = 49  # Set the number of EnKF filter-run loops
runtime = 0.1  # Set the runtime between each EnKF filtering

# Calculated Inputs
prog_endtime = init_runtime + num_loops * runtime


# ---------------------------------------------------------------------------------------------------
# Program run code

# Initialise solver source files for each member
print("\nINITIALISING CASE\n")
subprocess.run([sys.executable, "pythonScripts/initialise.py", str(num_members).strip(), str(mesh_num).strip(), str(init_runtime).strip(), str(file_write_freq).strip(), str(IC_type)])

# Process initial conditions for error calculation
members_array = [f"memberRunFiles/member{i}/VTK/member{i}_0.vtk" for i in range(1, num_members + 1)] + [exact_soln_path + str(0) + ".vtk"]
for sample_member in members_array: subprocess.run([sys.executable, "pythonScripts/VTKreadPV.py", sample_member])
subprocess.run([sys.executable, "pythonScripts/calcError.py", str(0)])

# Run solver for each member for the initial flow evolution/development stage - Uncomment only one 
print("\nRUNNING INTITIAL EVOLUTION OF MEMBERS\n")
# subprocess.run([sys.executable, "pythonScripts/runSolver.py"])  # Run each member 1 by 1
# subprocess.run([sys.executable, "pythonScripts/runSolverParallelPlot.py"])  # Run multiple ensemble members in parallel with one core each
subprocess.run([sys.executable, "pythonScripts/runSolverParallelPlot.py", str(init_runtime), str(0), str(prog_endtime)])  # Run multiple parallel members with progress plot

# Generate indices of cells for reduced resolution mesh
ref_timestep = int(init_runtime * 100)
subprocess.run([sys.executable, "pythonScripts/MeshResReducer.py", f"memberRunFiles/member1/VTK/member1_{ref_timestep}.vtk"])

# Reduce the resolution of the member results, as well as the exact solution. Write these to a directory XXXXXXXXXX in csv format
print("\nPROCESSING FIELDS\n")
members_array = [f"memberRunFiles/member{i}/VTK/member{i}_{ref_timestep}.vtk" for i in range(1, num_members + 1)] + [exact_soln_path + str(ref_timestep) + ".vtk"]
for sample_member in members_array: subprocess.run([sys.executable, "pythonScripts/VTKreadPV.py", sample_member])

# Calculate errors
subprocess.run([sys.executable, "pythonScripts/calcError.py", str(init_runtime)])

# Start Dapper Loop
print("------------------------------------------------")
print("\nSTARTING EnKF LOOP\n")
print("------------------------------------------------")
start_time = init_runtime
for loop_num in range(num_loops):
    print(f"\nDAPPER LOOP {loop_num+1}/{num_loops}\n")
    start_daploop_timing = time.time()
    subprocess.run([sys.executable, "pythonScripts/DAPPER.py"])
    print("\nWRITING NEW SOURCE FILES\n")
    subprocess.run([sys.executable, "pythonScripts/genNewICs.py", str(num_members), str(mesh_num), str(runtime), str(file_write_freq), str(start_time)])
    print("\nRUNNNING OPENFOAM CASES\n")
    subprocess.run([sys.executable, "pythonScripts/runSolverParallelPlot.py", str(start_time+runtime), str(start_time), str(prog_endtime)])  # Run multiple parallel members with progress plot
    print(f"\nPROCESSING FIELDS ({loop_num+1}/{num_loops})\n")
    members_array = [f"memberRunFiles/member{i}/VTK/member{i}_{int(round((start_time+runtime)*100))}.vtk" for i in range(1, num_members + 1)] + [exact_soln_path + str(int(round((start_time+runtime)*100))) + ".vtk"]
    for sample_member in members_array: subprocess.run([sys.executable, "pythonScripts/VTKreadPV.py", sample_member])
    subprocess.run([sys.executable, "pythonScripts/calcError.py", str(start_time+runtime)])
    start_time += runtime
    end_daploop_timing = time.time()
    print(f"\nDAPPER LOOP {loop_num+1}/{num_loops} FINISHED\nDapper loop {loop_num+1} elapsed time: {end_daploop_timing - start_daploop_timing:.2f} seconds\n")

print("\nSTARTING FINAL PROCESSING\n")

# Clean up directories to save storage - should now be redundant because source code changed to not output
subprocess.run([sys.executable, "pythonScripts/tidy.py"])

# Plot error metrics through time
subprocess.run([sys.executable, "pythonScripts/errorPlot.py"])

# Copy all .vtk files to outputs directory to allow for easily visualising in paraview
# subprocess.run([sys.executable, "pythonScripts/copyVisuals.py"])

# Automatically create .png files and .gif animations including the positions of sample points
# subprocess.run([sys.executable, "pythonScripts/animationMake2.py"])


# Time runtime 
end_whole_timing = time.time()
print(f"Whole program elapsed time: {end_whole_timing - start_whole_timing:.2f} seconds")

print("---------- END ----------")