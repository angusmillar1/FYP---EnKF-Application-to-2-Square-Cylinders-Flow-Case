# Generate the initial conditions based on the method defined in allrun
# This varies from the writing of new files each update since they are from the analysed field written
#   from the EnKF script

import numpy as np
import sys
import re
import os
import scipy.io
import random
import shutil

# Predefined inputs
num_members = int(sys.argv[1])      # Receive number of members from initialise.py script
meshNum = int(sys.argv[2])          # Receive mesh choice from initialise.py script
init_runtime = float(sys.argv[3])   # Receive the time for the members to initially evolve before informing from initialise.py
file_write_freq = int(sys.argv[4])  # Receive the frequency at which to write out data
IC_type = sys.argv[5]               # Receieve the method by which to initialise the ensemble

# Get number of cells in CFD domain based on the three predefined meshes in exampleOpenfoamFiles/Mesh{x}Files/Mesh/
num_cells = (
    18483 if meshNum == 1 else 
    66268 if meshNum == 2 else 
    84253 if meshNum == 3 else 
    None)
if num_cells is None: raise ValueError("Invalid meshNum value. Must be 1, 2, or 3.")

# Manually define flow quantities if uniform perturbed IC to be used (not recommended)
mean_u = 1.0            # Mean velocity magnitude
fluctuation_u = 0.68    # Velocity fluctuation range
mean_p = 0              # Mean pressure
fluctuation_p = 0.3     # Pressure fluctuation range

timeStep = "0"                                      # timeStep to generate field for
outputPath = "memberRunFiles/"                      # where to write sets of openfoam files
devInputPath = "exampleOpenfoamFiles/Mesh1DevICs"   # where to get previous simulation instances from if desired

# Get previosuly calculated POD data if POD-based initialisation desired
if IC_type == "POD":
    # Read in field statistics data
    mat = scipy.io.loadmat(f'inputs/podData/Mesh{str(meshNum)}_POD_Initialisation_data.mat')  # Load MATLAB data

    # Unpack variables from the .mat file
    Y_mean   = mat['Y_mean']      # Mean flow (stacked u,v)
    a_mean   = mat['a_mean']      # Mean of time coeffs 
    a_stddev = mat['a_stddev']    # Std dev of time coeffs 
    phi      = mat['phi_crop']    # Truncated mode matrix (stacked u',v')
    N, M = phi.shape  # N = number of spatial points/DOFs (rows of phi), M = number of retained modes (columns of phi)
            
# Helper function to generate a perturbed matrix given mean and fluctuation magnitudes
def generate_random_field(num_cells, mean, fluctuation):
    """
    Generate a random field with a specified mean and fluctuation range.
    """
    return mean + fluctuation * (np.random.rand(num_cells) - 0.5)

# Helper function to perturb the input matrix based on fluctuation magnitudes
def add_random_fluctuation(data, fluctuation):
    """
    Apply random fluctuations to a 2D vector (u,v) or a 1D vector p.
    """
    noise = np.random.uniform(-fluctuation, fluctuation, size=data.shape)
    return data + noise

# Helper function to read in OpenFOAM velocity vector data formatted exactly as defined
def read_U_data(filename):
    """
    Reads vector data from an OpenFOAM 0/U file with the specified format.
    """
    with open(filename, 'r') as file:
        content = file.read()

    # Extract the internalField block
    internal_field_match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+\d+\s+\((.*?)\)\s*;', content, re.DOTALL)
    if not internal_field_match:
        raise ValueError("Could not find the internalField block in the file.")

    vector_data_block = internal_field_match.group(1)

    # Extract individual vectors from the block
    vector_pattern = r'\((-?\d*\.?\d+(e[-+]?\d+)?)\s+(-?\d*\.?\d+(e[-+]?\d+)?)\s+(-?\d*\.?\d+(e[-+]?\d+)?)\)'
    vectors = re.findall(vector_pattern, vector_data_block)

    # Convert vectors to a numpy array
    vector_array = np.array([[float(x[0]), float(x[2])] for x in vectors])

    return vector_array

# Helper function to read in OpenFOAM pressure scalar data formatted exactly as defined
def read_p_data(filename):
    """
    Reads scalar data from an OpenFOAM file with the specified format.
    """
    with open(filename, 'r') as file:
        content = file.read()

    # Extract the internalField block
    internal_field_match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+\d+\s+\((.*?)\)\s*;', content, re.DOTALL)
    if not internal_field_match:
        raise ValueError("Could not find the internalField block in the file.")

    scalar_data_block = internal_field_match.group(1)

    # Extract individual scalar values from the block
    scalar_pattern = r'(-?\d*\.?\d+(e[-+]?\d+)?)'  # Matches float or scientific notation
    scalars = re.findall(scalar_pattern, scalar_data_block)

    # Convert scalars to a numpy array
    scalar_array = np.array([float(x[0]) for x in scalars])

    return scalar_array

# Helper function to write a velocity field OpenFOAM format file from a matrix of data
def write_U_file(filename, u_vectors, num_cells, timeStep):
    """
    Write the velocity field (U) file in the specified OpenFOAM format.
    """
    with open(filename, 'w') as f:
        f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       volVectorField;
    location    "{timeStep}";
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector> 
{num_cells}
(
""")
        for u in u_vectors:
            f.write(f"    ({u[0]:.3f} {u[1]:.3f} 0.000)\n")  # Limit to 6 decimal places
        f.write(f""");
boundaryField
{{
    INLET
    {{
        type            fixedValue;
        value           uniform (1 0 0);
    }}
    OUTLET
    {{
        type            zeroGradient;
    }}
    SYMMETRY
    {{
        type            symmetry;
    }}
    SQUARE_UP
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    SQUARE_DOWN
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    frontAndBackPlanes
    {{
        type            empty;
    }}
}}


// ************************************************************************* //
""")

# Helper function to write a pressure field OpenFOAM format file from a vector of data
def write_p_file(filename, p_values, num_cells, timeStep):
    """
    Write the pressure field (p) file in the specified OpenFOAM format.
    """
    with open(filename, 'w') as f:
        f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       volScalarField;
    location    "{timeStep}";
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar> 
{num_cells}
(
""")
        for p in p_values:
            f.write(f"    {p:.6f}\n")  # Limit to 6 decimal places
        f.write(f""");
boundaryField
{{
    INLET
    {{
        type            zeroGradient;
    }}
    OUTLET
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    SYMMETRY
    {{
        type            symmetry;
    }}
    SQUARE_UP
    {{
        type            zeroGradient;
    }}
    SQUARE_DOWN
    {{
        type            zeroGradient;
    }}
    frontAndBackPlanes
    {{
        type            empty;
    }}
}}


// ************************************************************************* //
""")

# Helper function to write a uniform zero pressure field OpenFOAM format file
def write_zero_p_file(filename, timeStep):
    """
    Write the pressure field (p) file in the specified OpenFOAM format.
    """
    with open(filename, 'w') as f:
        f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       volScalarField;
    location    "{timeStep}";
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    INLET
    {{
        type            zeroGradient;
    }}
    OUTLET
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    SYMMETRY
    {{
        type            symmetry;
    }}
    SQUARE_UP
    {{
        type            zeroGradient;
    }}
    SQUARE_DOWN
    {{
        type            zeroGradient;
    }}
    frontAndBackPlanes
    {{
        type            empty;
    }}
}}


// ************************************************************************* //
""")

# Helper function to write an OpenFOAM control file based on given inputs  
def write_controlDict_file(filename, init_runtime, file_write_freq):
    with open(filename, 'w') as f:
        f.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
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
""")

# Helper function to copy the set of files for the reference solution parallel run (or ensemble from developed IC)
def copy_reference_files(ref_destination_dir,randNum,meshNum):
    
    # Set path to find reference files
    source_dir = f"exampleOpenfoamFiles/Mesh{str(meshNum)}DevICs"

    # Copy pressure file
    shutil.copy(source_dir+"/p", ref_destination_dir)

    # Select and copy a random initial scene file from a set of pre-saved fields
    file_path = os.path.join(source_dir, "U", "U"+str(randNum))
    shutil.copy(file_path, ref_destination_dir+"U")



# Initialise random scene selection for reference (and members if used)
if IC_type == "prev": mem_randNum_set = set()   # Set to avoid duplicate members

numScenes = (
    100 if meshNum == 1 else    # 100 example fields saved for Mesh 1
    62 if meshNum == 2 else     # 62 example fields saved for Mesh 2
    100 if meshNum == 3 else    # 100 example fields saved for Mesh 3
    None)  
ref_randNum = random.randint(0,numScenes)  # Select random scene for the ground truth from the saved set

# Loop through each ensemble member
for memIndex in range(1, num_members+1):
    if IC_type == "rand":   # Randomly perturb a uniform field
        # Generate random fields
        u_magnitudes = generate_random_field(num_cells, mean_u, fluctuation_u)
        p_values = generate_random_field(num_cells, mean_p, fluctuation_p)
    
        # Generate random directions for velocity
        u_vectors = np.zeros((num_cells, 2))
        for i in range(num_cells):
            direction = np.random.rand(2) - 0.5         # Random direction
            direction /= np.linalg.norm(direction)      # Normalize to unit vector
            u_vectors[i] = u_magnitudes[i] * direction  # Scale by magnitude
    elif IC_type == "dev":  # Randomly perturb a single example previous field
        # Read in developed IC data
        u_inp = read_U_data(os.path.join(devInputPath, "U"))
        p_inp = read_p_data(os.path.join(devInputPath, "p"))

        # Apply random fluctuation
        u_vectors = add_random_fluctuation(u_inp, fluctuation_u)
        p_values = add_random_fluctuation(p_inp, fluctuation_p)
    elif IC_type == "POD":  # Generate random fields based on previous POD modes and time coeff stats
        # Generate randomised fields
        a_rand = a_mean + a_stddev * np.random.randn(M, 1)  # Draw a random set of time coefficients from normal distributions of each
        Y_rand = Y_mean + phi @ a_rand                      # Combine mean flow with mode-coefficient product pairs to get reduced order model fields
        u_vals = Y_rand[:num_cells, 0]                      # Extract the u values (1st half)
        v_vals = Y_rand[num_cells:, 0]                      # Extract the v values (2nd half)
        u_vectors = np.column_stack((u_vals, v_vals))       # Stack side-by-side
        p_values = np.zeros(num_cells)   # Just use zero pressure for now, should work out due to internal pimple loops
    elif IC_type == "prev": # Draw directly from a set of previously saved fields
        mem_randNum = random.randint(0,numScenes)   # Draw random field index
        while mem_randNum == ref_randNum or mem_randNum in mem_randNum_set: mem_randNum = random.randint(0,100) # Ensure no field is used twice in ensemble or ref soln
        mem_randNum_set.add(mem_randNum)            # Add to set so that index isn't used for the next member

        # Write all output files - different from other intialisation methods since U and p files copied not generated
        outputDir = outputPath + "member" + str(memIndex)
        copy_reference_files(outputDir+"/0/", mem_randNum, meshNum)
        write_controlDict_file(outputDir+"/system/controlDict", init_runtime, file_write_freq)
        print(f"Member{memIndex} files written using U{mem_randNum}.")
    else:
        raise ValueError("Invalid initial condition type selected")
    
    if IC_type != "prev":
        # Write to files
        outputDir = outputPath + "member" + str(memIndex)
        # print(outputDir)
        write_U_file(outputDir+"/0/U", u_vectors, num_cells, timeStep)
        # write_p_file(outputDir+"/0/p", p_values, num_cells, timeStep)
        write_zero_p_file(outputDir+"/0/p", timeStep)   # Zero pressure field used because it is instantly corrected, so it doesn't matter what values it takes
        write_controlDict_file(outputDir+"/system/controlDict", init_runtime, file_write_freq)
        print("Files 'U', 'p' and 'controlDict' have been generated.")

# Also do reference solution
copy_reference_files(outputPath+"refSoln/0/", ref_randNum, meshNum)   
write_controlDict_file(outputPath+"refSoln/system/controlDict", init_runtime, file_write_freq)
print(f"Reference solutions files written using U{ref_randNum}.")

