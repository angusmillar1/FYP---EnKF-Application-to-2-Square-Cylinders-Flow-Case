import os
import shutil
import subprocess
import sys
import re

# # Set the number of members
# num_members = 2  # Change this number as needed
# # Select the Mesh to use
# meshNum = 1

num_members = int(sys.argv[1])  # Receive number of members from ALLRUN.py script
mesh_num = int(sys.argv[2])  # Receive mesh choice from ALLRUN.py script
init_runtime = sys.argv[3]  # Receive the time for the members to initially evolve before informing from ALLRUN.py
file_write_freq = sys.argv[4]  # Receive the frequency at which to write out data
IC_type = sys.argv[5]

# Define the source and destination directories
source_dir = "exampleOpenfoamFiles/Mesh" + str(mesh_num) + "Files"
destination_parent_dir = "memberRunFiles"

# Delete all existing files and directories in the destination parent directory
if os.path.exists(destination_parent_dir):
    for item in os.listdir(destination_parent_dir):
        item_path = os.path.join(destination_parent_dir, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Remove the file or symbolic link
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the directory
    print(f"Cleared all contents in {destination_parent_dir}.")
else:
    os.makedirs(destination_parent_dir)
    print(f"Created directory {destination_parent_dir}.")

# Delete all reduced mesh files too
for file in os.listdir("EnKFMeshData/reducedMeshData"):
    if file.endswith(".csv"):  # Only target .csv files
        file_path = os.path.join("EnKFMeshData/reducedMeshData", file)
        os.remove(file_path)  # Delete the file
print("Cleared all contents in reducedMeshData")

for file in os.listdir("EnKFMeshData/fullMeshData"):
    if file.endswith(".csv"):  # Only target .csv files
        file_path = os.path.join("EnKFMeshData/fullMeshData", file)
        os.remove(file_path)  # Delete the file
print("Cleared all contents in fullMeshData")

for file in os.listdir("EnKFMeshData/filteredMeshData"):
    if file.endswith(".csv"):  # Only target .csv files
        file_path = os.path.join("EnKFMeshData/filteredMeshData", file)
        os.remove(file_path)  # Delete the file
print("Cleared all contents in filteredMeshData")

# Also delete past error data
def remove_files_in_directory(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):  # Check if it's a file
            os.remove(file_path)  # Remove the file
remove_files_in_directory("outputs")
remove_files_in_directory("outputs/visualisations")
print("Cleared all contents in outputs")

# Loop to copy and rename directories for each member
for i in range(1, num_members + 1):
    member_dir_name = f"member{i}"
    destination_dir = os.path.join(destination_parent_dir, member_dir_name)
    
    # Copy the source directory to the destination
    shutil.copytree(source_dir, destination_dir)
    print(f"Copied {source_dir} to {destination_dir}")

print("All member directories created successfully!")


# Generate initial conditions
subprocess.run([sys.executable, "pythonScripts/genFirstICs.py", str(num_members), str(mesh_num), init_runtime, file_write_freq, IC_type])

# Convert initial conditions to VTK files
member_directory = "memberRunFiles"
vtk_command = "bash -c '. /apps/openfoam/10.0/OpenFOAM-10/etc/bashrc && foamToVTK -time 0'"

# Regular expression to match directory names like "memberX" where X is a number
pattern = re.compile(r"^member\d+$")

# Step through each subdirectory in the main directory
for root, dirs, files in os.walk(member_directory):
    for directory in dirs:
        # Check if the directory name matches the pattern
        if pattern.match(directory):
            # Construct the full path to the subdirectory
            subdirectory_path = os.path.join(root, directory)

            # Change to the subdirectory
            print(f"Processing directory: {subdirectory_path}")
            try:
                # Execute the shell command in the subdirectory
                subprocess.run(vtk_command, cwd=subdirectory_path, shell=True,
                    stdout=open(os.path.join(subdirectory_path, "log.foamToVTK"), "w"),
                    stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(f"Command failed in directory {subdirectory_path}: {e}")

print("Initial conditions converted to readable output")


