import os
import shutil
import re

# Directories
member_run_files = "memberRunFiles"
# input_directory = "../referenceSolutions/Mesh1RandT500"
input_directory = "../referenceSolutions/Mesh1DevT500"
output_directory = "outputs/visualisations"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Regular expression to match .vtk filenames with the pattern *_xxx.vtk
vtk_pattern = re.compile(r".*_(\d+)\.vtk$")

# Collect the set of xxx numbers from the memberRunFiles
common_xxx_numbers = set()
for member_dir in os.listdir(member_run_files):
    member_path = os.path.join(member_run_files, member_dir, "VTK")
    if os.path.isdir(member_path):  # Ensure it's a valid directory
        for file in os.listdir(member_path):
            match = vtk_pattern.match(file)
            if match:
                common_xxx_numbers.add(match.group(1))  # Extract xxx

# Sort the xxx numbers for consistency (optional)
common_xxx_numbers = sorted(common_xxx_numbers)

# Copy .vtk files from memberRunFiles to outputs/visualisations
for member_dir in os.listdir(member_run_files):
    member_path = os.path.join(member_run_files, member_dir, "VTK")
    if os.path.isdir(member_path):  # Ensure it's a valid directory
        for file in os.listdir(member_path):
            match = vtk_pattern.match(file)
            if match and match.group(1) in common_xxx_numbers:
                src_file = os.path.join(member_path, file)
                dst_file = os.path.join(output_directory, file)
                shutil.copy2(src_file, dst_file)

# Copy matching .vtk files from input/ to outputs/visualisations
for file in os.listdir(input_directory):
    match = vtk_pattern.match(file)
    if match and match.group(1) in common_xxx_numbers:
        src_file = os.path.join(input_directory, file)
        dst_file = os.path.join(output_directory, file)
        shutil.copy2(src_file, dst_file)

print("All .vtk files have been copied successfully.")
