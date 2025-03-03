import numpy as np
import sys
import pandas as pd
import os
import csv

input_directory = "EnKFMeshData/fullMeshData"
timestep = float(sys.argv[1])
meshNum = sys.argv[2]
# timestep = 10

# Initialize lists to store data for members
mem_u = []
mem_v = []
# mem_p = []

# Read member files
Ne = 0
for filename in sorted(os.listdir(input_directory)):
    if filename.startswith("member") and filename.endswith(".csv"):
        filepath = os.path.join(input_directory, filename)
        data = pd.read_csv(filepath)
        mem_u.append(data['Ux'].values)
        mem_v.append(data['Uy'].values)
        # mem_p.append(data['p'].values)
        Ne += 1

# Convert lists to arrays with columns as data for each member
ens_u = np.column_stack(mem_u)
ens_v = np.column_stack(mem_v)
# ens_p = np.column_stack(mem_p)

# Read refSoln.csv into separate 1D arrays
ref_filepath = os.path.join(input_directory, "refSoln.csv")
ref_data = pd.read_csv(ref_filepath)
ref_u = ref_data['Ux'].values
ref_v = ref_data['Uy'].values
# ref_p = ref_data['p'].values

# Read cell volumes for integration
cellVols = np.loadtxt(f"inputs/cellVolumes/cellVolumes_mesh{meshNum}.txt")

print("Computing error norms (L1 and L2)")

# Compute the ensemble mean for each state
ens_u_mean = np.mean(ens_u, axis=1)
ens_v_mean = np.mean(ens_v, axis=1)
# ens_p_mean = np.mean(ens_p, axis=1)

# Compute errors (difference between ensemble mean and reference)
err_u = ens_u_mean - ref_u
err_v = ens_v_mean - ref_v
# err_p = ens_p_mean - ref_p

# Compute L1 norms (sum of absolute errors)
L1_u_dim = np.sum(np.abs(err_u)*cellVols)
L1_v_dim = np.sum(np.abs(err_v)*cellVols)
# L1_p = np.sum(np.abs(err_p))
# Combined L1 norm (treating the u and v errors as one long vector)
# L1_tot_dim = L1_u + L1_v

# Compute L2 norms (square root of sum of squared errors)
L2_u_dim = np.sqrt(np.sum((err_u**2) * cellVols))
L2_v_dim = np.sqrt(np.sum((err_v**2) * cellVols))
# L2_p = np.sqrt(np.sum(err_p ** 2))
# Combined L2 norm (for u and v together)
# L2_tot_dim = np.sqrt(np.sum(err_u ** 2 + err_v ** 2))
# Alternatively, if you prefer using the individual L2 values:
# L2_tot = np.sqrt(L2_u**2 + L2_v**2)

# Normalise norms by norm of reference solution
L1_u_ref = (np.sum(np.abs(ref_u)*cellVols))
L1_v_ref = (np.sum(np.abs(ref_v)*cellVols))
L2_u_ref = (np.sqrt(np.sum((ref_u ** 2)*cellVols)))
L2_v_ref = (np.sqrt(np.sum((ref_v ** 2)*cellVols)))

L1_u = L1_u_dim / L1_u_ref
L1_v = L1_v_dim / L1_v_ref
L2_u = L2_u_dim / L2_u_ref
L2_v = L2_v_dim / L2_v_ref

L1_tot = (np.sum((np.abs(err_u) + np.abs(err_v))*cellVols)) / (np.sum((np.abs(ref_u) + np.abs(ref_v))*cellVols))
L2_tot = (np.sqrt(np.sum(((err_u**2) + (err_v**2))*cellVols))) / (np.sqrt(np.sum(((ref_u**2) + (ref_v**2))*cellVols)))

# MSE
NP = len(ens_u_mean)
MSE_u = (1/NP) * (np.sum(err_u**2))
MSE_v = (1/NP) * (np.sum(err_v**2))
MSE_tot = (1/(2*NP)) * (np.sum((np.vstack((err_u,err_v)))**2))


# Write results to CSV file
output_file = "outputs/error_write.csv"

# Create the directory and file with headings if needed
if not os.path.exists(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Uncomment the following if you add pressure tracking later:
        # writer.writerow(["T", "L1_u", "L1_v", "L1_p", "L1_tot", "L2_u", "L2_v", "L2_p", "L2_tot"])
        writer.writerow(["T", "L1_u", "L1_v", "L1_tot", "L2_u", "L2_v", "L2_tot", "MSE_u", "MSE_v", "MSE_tot"])
        print(f"File '{output_file}' created with headings.")

# Append the calculated norms to the CSV file
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Uncomment the following if you add pressure tracking later:
    # writer.writerow([timestep, L1_u, L1_v, L1_p, L1_tot, L2_u, L2_v, L2_p, L2_tot])
    writer.writerow([timestep, L1_u, L1_v, L1_tot, L2_u, L2_v, L2_tot, MSE_u, MSE_v, MSE_tot])
    print(f"Values [T, L1_u, L1_v, L1_tot, L2_u, L2_v, L2_tot, MSE_u, MSE_v, MSE_tot] appended to error_write.csv")
