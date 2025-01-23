Dapper EnKF combined with OpenFoam implementation of flow reconstruction from sparse data



Outline of AllRun.py Code:
1. Call initialise
    - Copy example openfoam files for each member for the correct mesh, creating directories
    - Call genFirstICs.py to assign 0/U, 0/p and system/controlDict files to each member
        - Requires passing number of members between scripts via command line
    - 
2. Call runSolver.py
    - Run each member for an extended time to allow each system to develop end evolve before informing by EnKF
    - Convert output files to .vtk format
3. Call readVTK.py


