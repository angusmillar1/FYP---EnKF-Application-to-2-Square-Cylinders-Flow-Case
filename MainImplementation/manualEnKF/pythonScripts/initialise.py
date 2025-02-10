import os
import shutil
import subprocess
import sys
import re
import pandas as pd

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



# ------ DELETE DATA FROM PREVIOUS RUNS ------

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
remove_files_in_directory("outputs/visualisations/animations")
remove_files_in_directory("outputs/errorPlots")
print("Cleared all contents in outputs")







# ------ CREATE NEW FILES FOR NEW RUNS ------

# Loop to copy and rename directories for each member
for i in range(1, num_members + 1):
    member_dir_name = f"member{i}"
    destination_dir = os.path.join(destination_parent_dir, member_dir_name)
    
    # Copy the source directory to the destination
    shutil.copytree(source_dir, destination_dir)
    print(f"Copied {source_dir} to {destination_dir}")

# Copy reference solution run files
ref_destination_dir = os.path.join(destination_parent_dir, "refSoln")
shutil.copytree(source_dir, ref_destination_dir)
print(f"Copied {source_dir} to {ref_destination_dir}")

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
        if pattern.match(directory) or directory=="refSoln":
            # Construct the full path to the subdirectory
            subdirectory_path = os.path.join(root, directory)
            print(subdirectory_path)

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

# CREATE VELOCITY PROBES AT MEASUREMENT POINTS

# Generate indices of cells for reduced resolution mesh
subprocess.run([sys.executable, "pythonScripts/MeshResReducer.py", "memberRunFiles/member1/VTK/member1_0.vtk"])

# Rewrite controlDict files with probe points added
df = pd.read_csv("outputs/sample_points_locations.csv")
probe_points = df[['x', 'y', 'z']].values.tolist()

def write_controlDict_file(filename, init_runtime, file_write_freq, probe_points):
    header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     pimpleFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         {init_runtime};

deltaT          0.01; 

writeControl    timeStep;

writeInterval   {file_write_freq};

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

functions
{{
velocityProbes
{{
    type            probes;
    functionObjectLibs ("libsampling.so");
    outputControl   timeStep;       // Write every timestep
    outputInterval  1;
    probeLocations
    (
"""

    probe_lines = ""
    for point in probe_points:
        probe_lines += f"        ({point[0]:.3f} {point[1]:.3f} 0.000)\n"  # Limit to 6 decimal places   

    footer = f"""        );
    fields
    (
        U    // or (U p) if you need both velocity and pressure
    );
}}
Square_up
{{
    type        forceCoeffs;
    libs        ("libforces.so");
    log         no;
    patches     (SQUARE_UP);
    origin      (0 0 0);
    CofR                (0 0 0); // Centre of rotation
    dragDir             (1 0 0);
    liftDir             (0 1 0);    
    pitchAxis   (0 0 1);
    magUInf     1;
    rhoInf      1;
    rho         rhoInf;
    lRef        1;
    Aref        1;
}}
Square_down
{{
    type        forceCoeffs;
    libs        ("libforces.so");
    log         no;
    patches     (SQUARE_DOWN);
    origin      (0 0 0);
    CofR                (0 0 0); // Centre of rotation
    dragDir             (1 0 0);
    liftDir             (0 1 0);    
    pitchAxis   (0 0 1);
    magUInf     1;
    rhoInf      1;
    rho         rhoInf;
    lRef        1;
    Aref        1;
}}
     fieldAverage1
 {{
    // Mandatory entries (unmodifiable)
    type            fieldAverage;
    libs            ("libfieldFunctionObjects.so");

    // Mandatory entries (runtime modifiable)
    fields
    (
        U
        {{
            mean        yes;
            prime2Mean  no;
            base        time;
            windowType   exact;
            window       10000;
            windowName   <name>;
            allowRestart false;
        }}
        enstrophy
        {{
            mean        no;
            prime2Mean  no;
            base        time;
            windowType   exact;
            window       10000;
            windowName   <name>;
            allowRestart false;
        }}
        vorticity
        {{
            mean        no;
            prime2Mean  no;
            base        time;
            windowType   exact;
            window       10000;
            windowName   <name>;
            allowRestart false;
        }}        
               
    );

    // Optional entries (runtime modifiable)
    restartOnRestart    false;
    restartOnOutput     false;
    periodicRestart     false;
    restartPeriod       0.002;

    // Optional (inherited) entries
    region          region0;
    enabled         true;
    log             true;
    timeStart       50;
    timeEnd         10000;
    executeControl  timeStep;
    executeInterval 1;
    writeControl    timeStep;
    writeInterval   1000;
 }}
}}


// ************************************************************************* //
"""
    full_content = header + probe_lines + footer
    with open(filename, 'w') as f:
        f.write(full_content)

for memIndex in range(1, num_members+1):
    outputDir = "memberRunFiles/member" + str(memIndex)
    write_controlDict_file(outputDir+"/system/controlDict", init_runtime, file_write_freq, probe_points)

outputDir = "memberRunFiles/refSoln"
write_controlDict_file(outputDir+"/system/controlDict", init_runtime, file_write_freq, probe_points)




