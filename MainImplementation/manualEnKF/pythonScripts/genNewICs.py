import numpy as np
import sys
import os
import pandas as pd

def crop_number(num):
    num = round(num,1)
    if num.is_integer(): num = int(num)
    return num


# Predefined inputs
num_members = int(sys.argv[1])  # Receive number of members from initialise.py script
meshNum = int(sys.argv[2])  # Receive mesh choice from initialise.py script
runtime = float(sys.argv[3])  # Receive the time for the members to initially evolve before informing from initialise.py
file_write_freq = int(sys.argv[4])  # Receive the frequency at which to write out data
startTime = crop_number(float(sys.argv[5]))
endTime = startTime + runtime
# num_members = 1
# meshNum = 1
# runtime = 1
# file_write_freq = 100
# num_cells = 18483  # Number of cells ( 18483 / 66268 / 84253 )
num_cells = (
    18483 if meshNum == 1 else 
    66268 if meshNum == 2 else 
    84253 if meshNum == 3 else 
    None)
if num_cells is None: raise ValueError("Invalid meshNum value. Must be 1, 2, or 3.")


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
        for temp_u in u_vectors:
            u_val,v_val = temp_u
            f.write(f"    ({u_val:.5f} {v_val:.5f} 0.000)\n")  # Limit to 6 decimal places
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
        
def write_controlDict_file(filename, file_write_freq, startTime, endTime):
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

startTime       {startTime};

stopAt          endTime;

endTime         {endTime};

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

def main():

    for memIndex in range(1, num_members+1):

        # Read in mesh data
        inputDir = "EnKFMeshData/filteredMeshData"
        filename = "member" + str(memIndex) + ".csv"
        filepath = os.path.join(inputDir, filename)
        data = pd.read_csv(filepath)
        u = data['Ux'].values
        v = data['Uy'].values
        u_vectors = np.column_stack((u,v))
        p_values = data['p'].values

        
        # Write to files
        # outputDir = outputPath + "member" + str(memIndex)
        outputDir = "memberRunFiles/member" + str(memIndex)
        write_U_file(os.path.join(outputDir,str(startTime),"U"), u_vectors, num_cells, startTime)
        write_p_file(os.path.join(outputDir,str(startTime),"p"), p_values, num_cells, startTime)
        write_controlDict_file(outputDir+"/system/controlDict", file_write_freq, startTime, endTime)
        # print("Files 'U', 'p' and 'controlDict' have been generated.")

if __name__ == "__main__":
    main()
