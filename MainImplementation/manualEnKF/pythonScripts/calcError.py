import numpy as np
import sys
import pandas as pd
import os
import csv

# Select error metric outputs
mean_squared_error = 1
mean_absolute_percentage_error = 0
# etc

input_directory = "EnKFMeshData/fullMeshData"
timestep = float(sys.argv[1])

# Initialize lists to store data for members
mem_u = []
mem_v = []
mem_p = []

# Read member files
Ne = 0
for filename in sorted(os.listdir(input_directory)):
    if filename.startswith("member") and filename.endswith(".csv"):
        filepath = os.path.join(input_directory, filename)
        data = pd.read_csv(filepath)
        mem_u.append(data['Ux'].values)
        mem_v.append(data['Uy'].values)
        mem_p.append(data['p'].values)
        Ne += 1

# Convert lists to arrays with columns as data for each member
ens_u = np.column_stack(mem_u)
ens_v = np.column_stack(mem_v)
ens_p = np.column_stack(mem_p)

# Read refSoln.csv into separate 1D arrays
ref_filepath = os.path.join(input_directory, "refSoln.csv")
ref_data = pd.read_csv(ref_filepath)
ref_u = ref_data['Ux'].values
ref_v = ref_data['Uy'].values
ref_p = ref_data['p'].values

if mean_squared_error:
    print("Computing mean squared error")

    # Compute the ensemble mean
    ens_u_mean = np.mean(ens_u, axis=1)
    ens_v_mean = np.mean(ens_v, axis=1)
    ens_p_mean = np.mean(ens_p, axis=1)

    # Compute square errors
    se_u = (ens_u_mean - ref_u) ** 2
    se_v = (ens_v_mean - ref_v) ** 2
    se_p = (ens_p_mean - ref_p) ** 2

    # Compute MSE for each state and in total
    N = len(ens_u_mean)
    MSE_u = (1/N) * np.sum(se_u)
    MSE_v = (1/N) * np.sum(se_v)
    MSE_p = (1/N) * np.sum(se_p)
    MSE_tot = (1/(3*N)) * (np.sum(se_u) + np.sum(se_v) + np.sum(se_p))

    # Write results
    output_file = "outputs/error_write.csv"

    if not os.path.exists(output_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
        # Create the file with the specified headings
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["T", "MSE_u", "MSE_v", "MSE_p", "MSE_tot"])
            print(f"File '{output_file}' created with headings.")
    # else:
    #     print(f"File '{output_file}' already exists.")

    # Append the values to the file
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestep, MSE_u, MSE_v, MSE_p, MSE_tot])
        print(f"Values [T, MSE_u, MSE_v, MSE_p, MSE_tot] appended to error_write.csv")

