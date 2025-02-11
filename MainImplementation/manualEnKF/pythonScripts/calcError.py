import numpy as np
import sys
import pandas as pd
import os
import csv

input_directory = "EnKFMeshData/fullMeshData"
timestep = float(sys.argv[1])

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
L1_u = np.sum(np.abs(err_u))
L1_v = np.sum(np.abs(err_v))
# L1_p = np.sum(np.abs(err_p))
# Combined L1 norm (treating the u and v errors as one long vector)
L1_tot = L1_u + L1_v

# Compute L2 norms (square root of sum of squared errors)
L2_u = np.sqrt(np.sum(err_u ** 2))
L2_v = np.sqrt(np.sum(err_v ** 2))
# L2_p = np.sqrt(np.sum(err_p ** 2))
# Combined L2 norm (for u and v together)
L2_tot = np.sqrt(np.sum(err_u ** 2 + err_v ** 2))
# Alternatively, if you prefer using the individual L2 values:
# L2_tot = np.sqrt(L2_u**2 + L2_v**2)

# Normalise norms by norm of reference solution
L1_ref_u = (np.sum(np.abs(ref_u)))
L1_ref_v = (np.sum(np.abs(ref_v)))
L2_ref_u = (np.sqrt(np.sum(ref_u ** 2)))
L2_ref_v = (np.sqrt(np.sum(ref_v ** 2)))

L1_u = L1_u / L1_ref_u
L1_v = L1_v / L1_ref_v
L2_u = L2_u / L2_ref_u
L2_v = L2_v / L2_ref_v

L1_tot = ((L1_u * L1_ref_u) + (L1_v * L1_ref_v)) / (L1_ref_u + L1_ref_v)
L2_tot = (np.sqrt((L2_u*L2_ref_u)**2 + (L2_v*L2_ref_v)**2)) / (np.sqrt(L2_ref_u**2 + L2_ref_v**2))

# Write results to CSV file
output_file = "outputs/error_write.csv"

# Create the directory and file with headings if needed
if not os.path.exists(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Uncomment the following if you add pressure tracking later:
        # writer.writerow(["T", "L1_u", "L1_v", "L1_p", "L1_tot", "L2_u", "L2_v", "L2_p", "L2_tot"])
        writer.writerow(["T", "L1_u", "L1_v", "L1_tot", "L2_u", "L2_v", "L2_tot"])
        print(f"File '{output_file}' created with headings.")

# Append the calculated norms to the CSV file
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Uncomment the following if you add pressure tracking later:
    # writer.writerow([timestep, L1_u, L1_v, L1_p, L1_tot, L2_u, L2_v, L2_p, L2_tot])
    writer.writerow([timestep, L1_u, L1_v, L1_tot, L2_u, L2_v, L2_tot])
    print(f"Values [T, L1_u, L1_v, L1_tot, L2_u, L2_v, L2_tot] appended to error_write.csv")
