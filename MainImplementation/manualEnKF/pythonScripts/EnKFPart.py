import pandas as pd
import numpy as np
import os
import statistics as st
import re
# import sys
# import csv

# def crop_number(num):
#     num = round(num,1)
#     if num.is_integer(): num = int(num)
#     return num

def generate_R(N,sigma,rho):
    R = np.full((N, N), rho * sigma**2)
    np.fill_diagonal(R, sigma**2)
    return R

def generate_perturbations_cholesky(R, P):
    N = R.shape[0]
    # Compute Cholesky factor L, so that R = L * L.T
    L = np.linalg.cholesky(R)
    
    # Preallocate array for the samples
    epsilons = np.zeros((N, P))
    
    for i in range(P):
        # Draw a standard normal vector z (N-dim)
        z = np.random.randn(N)
        # Transform it: epsilon = L @ z has covariance R
        epsilons[:, i] = L @ z
    
    return epsilons

def enkf_update_vectorized(ens_u, ens_v, u_true, v_true, 
                           sigma_u, sigma_v, 
                           rho_u,  rho_v):
    """
    Perform one EnKF update step (analysis) for separate u,v velocity components.
    Treat all spatial points at once (no loop over points).
    
    Parameters
    ----------
    ens_u : ndarray, shape (N_p, N_e)
        Forecast ensemble for the u-component. 
        Each row = one spatial point, each column = one ensemble member.
    ens_v : ndarray, shape (N_p, N_e)
        Forecast ensemble for the v-component, same shape as ens_u.
    u_true : ndarray, shape (N_p,)
        "True" or reference observation for the u-component at each spatial point.
    v_true : ndarray, shape (N_p,)
        "True" or reference observation for the v-component at each spatial point.
    sigma_u, sigma_v : floats
        Standard deviations of measurement noise for u, v.
    rho_u, rho_v : floats
        Cross-point noise correlation for u, v (0 => diagonal R).
    
    Returns
    -------
    ens_u_an, ens_v_an : ndarrays, shape (N_p, N_e)
        The updated (analysis) ensembles for u and v.
    """
    N_p, N_e = ens_u.shape  # number of points, number of ensemble members
    
    # (1) Build the R (covariance) matrices for u and v
    R_u = generate_R(N_p, sigma_u, rho_u)  # shape (N_p, N_p)
    R_v = generate_R(N_p, sigma_v, rho_v)  # shape (N_p, N_p)
    
    # (2) Generate random measurement-error perturbations for each ensemble member
    #     Each of these will be shape (N_p, N_e)
    eps_u = generate_perturbations_cholesky(R_u, N_e)
    eps_v = generate_perturbations_cholesky(R_v, N_e)
    
    # (3) Construct the "perturbed observations" for each ensemble member
    #     We'll treat each column as a distinct random draw of the observation.
    #     u_true, v_true are shape (N_p,). We tile them into NxP, then add epsilons.
    Y_u = np.tile(u_true, (N_e, 1)).T + eps_u  # shape (N_p, N_e)
    Y_v = np.tile(v_true, (N_e, 1)).T + eps_v  # shape (N_p, N_e)

    # (4) Compute the forecast mean for each velocity component
    #     shape (N_p,)
    u_mean = np.mean(ens_u, axis=1)
    v_mean = np.mean(ens_v, axis=1)
    
    # (5) Compute forecast anomalies (deviations from the mean)
    #     ens_u_ano, ens_v_ano are (N_p, N_e)
    ens_u_ano = ens_u - u_mean[:, None]
    ens_v_ano = ens_v - v_mean[:, None]
    
    # (6) Sample forecast covariance for each velocity field
    #     shape (N_p, N_p)
    cov_u = (1.0 / (N_e - 1.0)) * ens_u_ano @ ens_u_ano.T
    cov_v = (1.0 / (N_e - 1.0)) * ens_v_ano @ ens_v_ano.T
    
    # (7) Compute the Kalman gain:
    #     K_u, K_v are shape (N_p, N_p)
    #     K = P_f (P_f + R)^{-1}
    K_u = cov_u @ np.linalg.inv(cov_u + R_u)
    K_v = cov_v @ np.linalg.inv(cov_v + R_v)
    
    # (8) Apply the EnKF update to each ensemble member:
    #     X_a = X_f + K ( Y - X_f )
    #     We can do this in a fully vectorized way:
    #       - (Y - ens_u) is (N_p, N_e)
    #       - K_u is (N_p, N_p), so K_u @ (Y - ens_u) is also (N_p, N_e)
    ens_u_an = ens_u + K_u @ (Y_u - ens_u)
    ens_v_an = ens_v + K_v @ (Y_v - ens_v)
    
    return ens_u_an, ens_v_an


# # Take start time for error tracking write
# timestep = crop_number(float(sys.argv[5]))

# Directory containing the .csv files
redu_directory = "EnKFMeshData/reducedMeshData"
full_directory = "EnKFMeshData/fullMeshData"
outp_directory = "EnKFMeshData/filteredMeshData/"

# Initialize lists to store data for members
Ux_data = []
Uy_data = []
# p_data = []
IDs = []

# (1) Read member files
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




# (2) IMPLEMENT KALMAN FILTERING MANUALLY
sigma_u, sigma_v = 0.1, 0.05    # std dev in measurement noise
rho_u, rho_v = 0.0, 0.0         # Correlations in measurement noise (0 for now)
Np = len(ens_u)                 # Number of points

# Generate measurement noise matrices
R_u = generate_R(Np,sigma_u,rho_u)
R_v = generate_R(Np,sigma_v,rho_v)

# Add random perturbations to truth values
epsilons_u = generate_perturbations_cholesky(R_u, Ne)
epsilons_v = generate_perturbations_cholesky(R_v, Ne)

# Call EnKF update function
ens_u, ens_v = enkf_update_vectorized(
        ens_u, ens_v, ref_u, ref_v,
        sigma_u, sigma_v, rho_u, rho_v
    )

# print("Updated ensemble (u):", ens_u.shape)


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


# # WRITE DATA FOR MEASUREMENT POINT ERROR TRACKING - EDIT FROM HERE DOWN

# # Write results to CSV file
# point_data_output_file = "outputs/measurment_points_track.csv"

# # Create the directory and file with headings if needed
# if not os.path.exists(point_data_output_file):
#     os.makedirs(os.path.dirname(point_data_output_file), exist_ok=True)
#     with open(point_data_output_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["T", "ens_u_max", "ens_u_min", "ens_u_mean", "ens_v_max", "ens_v_min", "ens_v_mean","ref_u","ref_v"])
#         print(f"File '{point_data_output_file}' created with headings.")

# # Append the calculated norms to the CSV file
# with open(point_data_output_file, mode='a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([timestep, ens_u_max, ens_u_min, ens_u_mean, ens_v_max, ens_v_min, ens_v_mean, ref_u, ref_v])
#     print(f"Values [T, L1_u, L1_v, L1_tot, L2_u, L2_v, L2_tot] appended to error_write.csv")