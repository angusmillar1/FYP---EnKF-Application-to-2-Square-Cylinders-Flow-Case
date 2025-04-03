import os
import subprocess
import sys
import time
import matplotlib.pyplot as plt

start_whole_timing = time.time()  # Time runtime

# PREDEFINED INPUTS

# Initial parameters
mesh_num = 2  # Select the Mesh to use
file_write_freq = 500  # Frequency at which to write out data, assuming deltaT=0.01 (100=>T=1)
IC_type = "POD"  # "rand" / "dev" / "POD" / "prev". Define initial conditiion type to use: either random, developed, POD-based, or previous solution
exact_soln_path = "memberRunFiles/refSoln/VTK/refSoln_"  # Runs in parallel with ensemble members now, shouldn't ever have to change
# init_runtime = 5  # Set the time for the members to initially evolve before informing (commented if same as runtime)

# Ensemble and filtering parameters
num_members = 5     # Set the number of ensemble members
runtime = 25         # Set the runtime between each EnKF filtering
prog_endtime = 400   # Set the total run time of the program

# Calculated Inputs
init_runtime = runtime   # Comment if different initial runtime required - unlikely
num_loops = (prog_endtime - init_runtime)/runtime   # Determine the number of EnKF filter-run loops
if int(num_loops) != num_loops: print("\n !!!!! WARNING - INVALID NUMBER OF LOOPS !!!!! \n"); sys.exit()
if int((runtime*100)/file_write_freq) != ((runtime*100)/file_write_freq): print("\n !!!!! WARNING - WRITE FREQUENCY MUST DIVIDE RUN TIME !!!!! \n"); sys.exit()
num_loops = int(num_loops)



# ---------------------------------------------------------------------------------------------------
# Program run code

# Initialise solver source files for each member
print("\nINITIALISING CASE\n")
subprocess.run([sys.executable, "pythonScripts/initialise.py", str(num_members).strip(), str(mesh_num).strip(), str(init_runtime).strip(), str(file_write_freq).strip(), str(IC_type)])

# Process initial conditions for error calculation
members_array = [f"memberRunFiles/member{i}/VTK/member{i}_0.vtk" for i in range(1, num_members + 1)] + [exact_soln_path + str(0) + ".vtk"]
for sample_member in members_array: subprocess.run([sys.executable, "pythonScripts/VTKreadPV.py", sample_member])
subprocess.run([sys.executable, "pythonScripts/calcError.py", str(0), str(mesh_num)])

# Run solver for each member for the initial flow evolution/development stage - Uncomment only one 
print("\nRUNNING INTITIAL EVOLUTION OF MEMBERS\n")
# subprocess.run([sys.executable, "pythonScripts/runSolver.py"])  # Run each member 1 by 1
# subprocess.run([sys.executable, "pythonScripts/runSolverParallelPlot.py"])  # Run multiple ensemble members in parallel with one core each
subprocess.run([sys.executable, "pythonScripts/runSolverParallelPlot.py", str(init_runtime), str(0), str(prog_endtime)])  # Run multiple parallel members with progress plot

# Reduce the resolution of the member results, as well as the exact solution. Write these to a directory XXXXXXXXXX in csv format
print("\nPROCESSING FIELDS\n")
ref_timestep = int(init_runtime*100)
members_array = [f"memberRunFiles/member{i}/VTK/member{i}_{ref_timestep}.vtk" for i in range(1, num_members + 1)] + [exact_soln_path + str(ref_timestep) + ".vtk"]
for sample_member in members_array: subprocess.run([sys.executable, "pythonScripts/VTKreadPV.py", sample_member])

# Calculate errors
subprocess.run([sys.executable, "pythonScripts/calcError.py", str(init_runtime), str(mesh_num)])

# Start EnKF Loop
print("------------------------------------------------")
print("\nSTARTING EnKF LOOP\n")
print("------------------------------------------------")
start_time = init_runtime
for loop_num in range(num_loops):
    print(f"\nEnKF LOOP {loop_num+2}/{num_loops+1}\n")
    start_daploop_timing = time.time()
    subprocess.run([sys.executable, "pythonScripts/EnKFFull2.py"])
    print("\nWRITING NEW SOURCE FILES\n")
    subprocess.run([sys.executable, "pythonScripts/genNewICs.py", str(num_members), str(mesh_num), str(runtime), str(file_write_freq), str(start_time)])
    print("\nRUNNNING OPENFOAM CASES\n")
    subprocess.run([sys.executable, "pythonScripts/runSolverParallelPlot.py", str(start_time+runtime), str(start_time), str(prog_endtime)])  # Run multiple parallel members with progress plot
    print(f"\nPROCESSING FIELDS ({loop_num+2}/{num_loops+1})\n")
    members_array = [f"memberRunFiles/member{i}/VTK/member{i}_{int(round((start_time+runtime)*100))}.vtk" for i in range(1, num_members + 1)] + [exact_soln_path + str(int(round((start_time+runtime)*100))) + ".vtk"]
    for sample_member in members_array: 
        result1 = subprocess.run([sys.executable, "pythonScripts/VTKreadPV.py", sample_member], capture_output=True, text=True)
        if result1.returncode != 0: break
    result2 = subprocess.run([sys.executable, "pythonScripts/calcError.py", str(start_time+runtime), str(mesh_num)])
    if result2.returncode != 0: break
    start_time += runtime
    end_daploop_timing = time.time()
    print(f"\nEnKF LOOP {loop_num+2}/{num_loops+1} FINISHED\nEnKF loop {loop_num+2} elapsed time: {((end_daploop_timing - start_daploop_timing)/60):.2f} minutes\n")

print("\nSTARTING FINAL PROCESSING\n")

# Clean up directories to save storage - should now be redundant because source code changed to not output
subprocess.run([sys.executable, "pythonScripts/tidy.py"])

# Plot error metrics through time
print("Creating error plots")
subprocess.run([sys.executable, "pythonScripts/errorPlot.py", str(num_members), str(runtime)])

# Copy all .vtk files to outputs directory to allow for easily visualising in paraview
print("Moving vtk files")
subprocess.run([sys.executable, "pythonScripts/copyVisuals.py"])

# Automatically create .png files and .gif animations including the positions of sample points
print("Creating animations")
subprocess.run([sys.executable, "pythonScripts/animationMake2.py"])

# Notify of completion
subprocess.run("echo 'The run has finished!' | mail -s 'Job Done' acm21@ic.ac.uk", shell=True)

# Time runtime 
end_whole_timing = time.time()
print(f"Whole program elapsed time: {((end_whole_timing - start_whole_timing)/3600):.2f} hours")

print("---------- END ----------")