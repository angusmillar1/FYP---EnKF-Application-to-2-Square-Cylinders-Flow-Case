import pandas as pd
import numpy as np
import os
import statistics as st
import re

# Directory containing the .csv files
redu_directory = "EnKFMeshData/reducedMeshData"
full_directory = "EnKFMeshData/fullMeshData"
outp_directory = "EnKFMeshData/filteredMeshData/"

# Initialize lists to store data for members
Ux_data = []
Uy_data = []
# p_data = []
IDs = []

# Read member files
Ne = 0
for filename in sorted(os.listdir(redu_directory)):
    if filename.startswith("member") and filename.endswith(".csv"):
        filepath = os.path.join(redu_directory, filename)
        data = pd.read_csv(filepath)
        Ux_data.append(data['Ux'].values)
        Uy_data.append(data['Uy'].values)
        # p_data.append(data['p'].values)
        IDs.append(data['CellID'].values)
        Ne += 1

# Convert lists to arrays with columns as data for each member
ens_u = np.column_stack(Ux_data)
ens_v = np.column_stack(Uy_data)
# ens_p = np.column_stack(p_data)

# Read refSoln.csv into separate 1D arrays
ref_filepath = os.path.join(redu_directory, "refSoln.csv")
ref_data = pd.read_csv(ref_filepath)
ref_u = ref_data['Ux'].values
ref_v = ref_data['Uy'].values
# ref_p = ref_data['p'].values
ref_IDs = ref_data['CellID'].values




# IMPLEMENT KALMAN FILTERING MANUALLY
obs_noise = 0.000001
Np = len(ens_u)

for i in range(Np):  # Loop through for each sample point

    # Compute the ensemble mean
    ens_u_mean = st.mean(ens_u[i,:])
    ens_v_mean = st.mean(ens_v[i,:])
    # ens_p_mean = st.mean(ens_p[i,:])

    # Compute the sample covariance between ensemble predictions and observations
    cov_u = (1 / (Ne - 1)) * np.sum((ens_u - ens_u_mean) * (ref_u[i] - ens_u_mean))
    cov_v = (1 / (Ne - 1)) * np.sum((ens_v - ens_v_mean) * (ref_v[i] - ens_v_mean))
    # cov_p = (1 / (Ne - 1)) * np.sum((ens_p - ens_p_mean) * (ref_p[i] - ens_p_mean))

    # Compute the variances of ensemble predictions
    var_u = (1 / (Ne - 1)) * np.sum((ens_u - ens_u_mean)**2)
    var_v = (1 / (Ne - 1)) * np.sum((ens_v - ens_v_mean)**2)
    # var_p = (1 / (Ne - 1)) * np.sum((ens_p - ens_p_mean)**2)

    # Compute the Kalman gain (scalar value for each point)
    K_u = cov_u / (var_u + obs_noise**2)
    K_v = cov_v / (var_v + obs_noise**2)
    # K_p = cov_p / (var_p + obs_noise**2)

    for j in range (Ne):  # Loop through each member to write new values to array
        ens_u[i,j] += K_u * (ref_u[i] - ens_u[i,j])
        ens_v[i,j] += K_v * (ref_v[i] - ens_v[i,j])
        # ens_p[i,j] += K_p * (ref_p[i] - ens_p[i,j])




# WRITE NEW FILES

# Create arrays to read full data into
Ux_data_full = []
Uy_data_full = []
# p_data_full = []
IDs_full = []

# Loop only once for each member
for filename in sorted(os.listdir(full_directory)):
    if filename.startswith("member") and filename.endswith(".csv"):
        filepath = os.path.join(full_directory, filename)
        data = pd.read_csv(filepath)
        Ux_data_full = data['Ux'].values
        Uy_data_full = data['Uy'].values
        # p_data_full = data['p'].values
        IDs_full = data['CellID'].values
        # Get current member number
        match = re.search(r"member(\d+)\.csv", filename)
        if match: mem_num = int(match.group(1))

        # Initialise write vars
        # u_write, v_write, p_write = [np.empty(len(IDs_full)) for _ in range(3)]
        u_write, v_write = [np.empty(len(IDs_full)) for _ in range(2)]

        # Create a dictionary for fast lookup of indices in ref_IDs
        ref_IDs_dict = {id_: idx for idx, id_ in enumerate(ref_IDs)}

        # Loop through IDs_full and process
        for i, id_ in enumerate(IDs_full):
            if id_ in ref_IDs_dict:
                redu_index = ref_IDs_dict[id_]
                u_write[i] = ens_u[redu_index, mem_num-1]
                v_write[i] = ens_v[redu_index, mem_num-1]
                # p_write[i] = ens_p[redu_index, mem_num-1]
            else:
                u_write[i] = Ux_data_full[i]
                v_write[i] = Uy_data_full[i]
                # p_write[i] = p_data_full[i]

        # Create a DataFrame
        export_data = {
            "Ux": u_write,
            "Uy": v_write#,
            # "p": p_write
            }
        df = pd.DataFrame(export_data)

        # Write to CSV
        output_file = outp_directory + "member" + str(mem_num) + ".csv"
        df.to_csv(output_file, index=False, float_format="%.6f")

        # print(f"Filtered data written for member " + str(mem_num))
