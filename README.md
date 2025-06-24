Summary
---
This is my implementation of the application of the ensemble kalman filter algorithm to the Re=200, g*=1, 2D, two-square-cylinders flow case.
The code is spread across multiple scripts contained within pythonScripts/, which are all called in turn from ALLRUN.py. The structure is as follows, where indentation refers to scripts called from within the one above.

Code Structure
---
- ALLRUN.py - Take inputs, call all other scripts in order
  - initialise.py - Delete files from previous runs, create new file structure
      - genFirstICs.py - Generate the first set of initial conditions
      - MeshResReducer.py - Get the positions of measurement points
  - runSolverParallelPlot.py - Run DNS solutions in parallel and plot progress
  - VTKreadPV.py - Read .vtk OpenFOAM output files into .csv and extract data at measurement points
  - calcError.py - Calculate full field summary error metrics
  *** start loop ***
      - EnKFFull2.py - Analyse and update samples as per the Kalman filter equations
      - genNewICs.py - Write the updated fields to OpenFOAM files
      - runSolverParallelPlot.py - Run DNS solutions in parallel and plot progress
      - VTKreadPV.py - Read .vtk OpenFOAM output files into .csv and extract data at measurement points
      - calcError.py - Calculate full field summary error metrics
  *** end loop
  - tidy.py - Remove large redundant files
  - errorPlot.py - Plot error metrics and other analysis quantities
  - copyVisuals.py - Collate simulation output .vtk files for comparison
  - animationMake2.py - Generate .png and .gif versions of the evolutions of states

Directory Structure
---
- MainImplementation/						- Parent folder 
	- EnKFMeshData/									- Program writes .csv files here
		- filteredMeshData/							- post-update data written after update
		- fullMeshData/									- pre-update data from whole domain
		- postUpdateFullMeshData/ 			- post-update data from previous update
		- reducedMeshData/							- pre-update data at measurement points
	- exampleOpenfoamFiles/					- Run files to copy from
 		- Mesh1Files/										- Coarse mesh source files
	 	- Mesh2Files/										- Medium mesh source files
	 	- Mesh3Files/  									- Fine mesh source files
	 	- Mesh1DevICs/									- Set of stored previous fields for ICs
	 	- Mesh2DevICs/									- Set of stored previous fields for ICs
	- inputs/ 											- User input data 
 		- cellCentres/									- Cell centre locations for each mesh
	 	- cellVolumes/									- Cell volumes for each mesh (area since unit depth)
	 	- mesurementPoints/							- Desired measurment points (inc. .py to make them)
	 	- podData/											- .mats containing POD eigenmode shapes, mean flow and time coefficient stats
	- memberRunFiles/								- Runtime OpenFOAM source dirs
 	- outputs/											- Error metric plots and visualisations
	- pythonScripts/								- All code for running the program

Case Adaptation
---
This code can be adapted to other geometries, flow conditions or meshes, but will require significant adjustments.
First, the OpenFOAM source files should be built and written to exampleOpenfoamFiles/, including meshes in .msh files within.
Other related inputs such as cell centres and volumes should also be written to inputs/.
All the scripts should then be adjusted to ensure that everything is referenced correctly, particularly is alternative solvers or meshes are used.
That being said, much of the structure and logic contained here should be applicable to variations, and it should just be a case of checking each line still works as intended.

Notes
---
This code should be used with multiple caveats. It was designed to work on the university linux machines, which have tight restrictions on the installation of packages, hence the required python packages were downloaded from their respective repos and included manually in pythonScripts/.
This code is also far from perfect, and was designed (by an engineer) to perform the required task with satisfactory efficiency and is far from fully optimised.
