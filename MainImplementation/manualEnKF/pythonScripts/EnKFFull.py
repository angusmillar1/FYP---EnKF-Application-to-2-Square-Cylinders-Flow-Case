import pandas as pd
import numpy as np
import os
import re

# ----------------------------
# Helper functions
# ----------------------------
def generate_R(N, sigma, rho):
    """
    Generate an N x N measurement noise covariance matrix.
    Diagonal: sigma^2; off-diagonals: rho*sigma^2.
    """
    R = np.full((N, N), rho * sigma**2)
    np.fill_diagonal(R, sigma**2)
    return R

def generate_perturbations_cholesky(R, P):
    """
    Generate P samples (columns) from N(0,R) using a Cholesky factorization.
    R must be symmetric positive definite.
    """
    N = R.shape[0]
    L = np.linalg.cholesky(R)
    epsilons = np.zeros((N, P))
    for i in range(P):
        z = np.random.randn(N)
        epsilons[:, i] = L @ z
    return epsilons

def enkf_update_full_field(ens, y, measured_idx, sigma, rho):
    """
    Perform an EnKF update for the full field.
    
    Parameters
    ----------
    ens : ndarray, shape (N_full, N_e)
        The forecast ensemble for a given field (e.g. Ux).
        Each column is one ensemble member.
    y : ndarray, shape (P,)
        The observation vector. These are the "true" values at the measured
        locations (the reduced field). P is the number of measured points.
    measured_idx : array-like of length P
        Indices into the full field that are measured.
    sigma : float
        The standard deviation of measurement noise.
    rho : float
        The correlation (off-diagonal) parameter for the measurement noise.
    
    Returns
    -------
    ens_a : ndarray, shape (N_full, N_e)
        The analysis (updated) ensemble for the full field.
    
    Notes
    -----
    The measurement operator H is implemented implicitly by extracting the rows
    of the state vector corresponding to measured indices.
    """
    N_full, N_e = ens.shape
    P = len(measured_idx)
    
    # (a) Measurement noise covariance for measured space (P x P)
    R = generate_R(P, sigma, rho)
    
    # (b) Generate measurement error perturbations: shape (P, N_e)
    eps = generate_perturbations_cholesky(R, N_e)
    
    # (c) Form the perturbed observations for each ensemble member:
    #     y has shape (P,) so we tile it into a (P, N_e) array and add eps.
    Y = np.tile(y, (N_e, 1)).T + eps   # shape (P, N_e)
    
    # (d) Compute the full–state ensemble mean and anomalies.
    x_mean = np.mean(ens, axis=1)        # shape (N_full,)
    X_ano  = ens - x_mean[:, None]       # shape (N_full, N_e)
    
    # (e) In measurement space, H simply extracts the rows indexed by measured_idx.
    #     Compute the measurement anomalies:
    S = X_ano[measured_idx, :]           # shape (P, N_e)
    
    # (f) Compute the innovation covariance in measurement space:
    #     cov_yy = (1/(N_e-1))*S S^T + R
    cov_yy = (1.0/(N_e - 1)) * (S @ S.T) + R   # shape (P, P)
    
    # (g) Compute the cross-covariance between full state and measurement space:
    #     P_xy = (1/(N_e-1))* X_ano * S^T
    P_xy = (1.0/(N_e - 1)) * (X_ano @ S.T)       # shape (N_full, P)
    
    # (h) Compute the Kalman gain:
    K = P_xy @ np.linalg.inv(cov_yy)             # shape (N_full, P)
    
    # (i) Get the forecasted measurement for each ensemble member:
    Hxf = ens[measured_idx, :]                   # shape (P, N_e)
    
    # (j) EnKF update: add the innovation (Y - Hxf) back to the full state:
    ens_a = ens + K @ (Y - Hxf)                  # shape (N_full, N_e)
    
    return ens_a

# ----------------------------
# Main code: reading data and updating the full field
# ----------------------------

# Directories:
# - The reduced directory contains the measurements (refSoln.csv) with the measured cell IDs.
# - The full directory contains the full–field ensemble for each member.
redu_directory = "EnKFMeshData/reducedMeshData"
full_directory = "EnKFMeshData/fullMeshData"
outp_directory = "EnKFMeshData/filteredMeshData/"

# (1) Read the reference (measurement) data from the reduced mesh.
ref_filepath = os.path.join(redu_directory, "refSoln.csv")
ref_data = pd.read_csv(ref_filepath)
ref_u   = ref_data['Ux'].values
ref_v   = ref_data['Uy'].values
ref_IDs = ref_data['CellID'].values

# (2) Read the full–field ensemble from the full directory.
full_ens_u_list = []
full_ens_v_list = []
IDs_full = None

# Get all member files from the full field directory.
member_filenames = sorted([f for f in os.listdir(full_directory) if f.startswith("member") and f.endswith(".csv")])
Ne = len(member_filenames)
for filename in member_filenames:
    filepath = os.path.join(full_directory, filename)
    data = pd.read_csv(filepath)
    # Assume each CSV has columns "Ux", "Uy", "CellID"
    if IDs_full is None:
        IDs_full = data['CellID'].values  # full field cell IDs
    full_ens_u_list.append(data['Ux'].values)
    full_ens_v_list.append(data['Uy'].values)

# Convert lists into ensemble arrays.
# Each column corresponds to one ensemble member.
ens_u_full = np.column_stack(full_ens_u_list)  # shape: (N_full, Ne)
ens_v_full = np.column_stack(full_ens_v_list)  # shape: (N_full, Ne)

# (3) Determine which cells in the full field are measured.
# Build a dictionary mapping full field cell IDs to their index.
full_IDs_dict = {id_: idx for idx, id_ in enumerate(IDs_full)}
# For each cell ID in the reduced (measurement) data, find its index in the full field.
measured_idx = [full_IDs_dict[id_] for id_ in ref_IDs if id_ in full_IDs_dict]
measured_idx = np.array(measured_idx)  # shape: (P,)

# (4) Set measurement noise parameters.
sigma_u, sigma_v = 0.1, 0.05    # standard deviations for Ux and Uy
rho_u, rho_v     = 0.0, 0.0      # assume no cross–point correlation for now

# (5) Update the full field ensemble using the EnKF update.
ens_u_full_updated = enkf_update_full_field(ens_u_full, ref_u, measured_idx, sigma_u, rho_u)
ens_v_full_updated = enkf_update_full_field(ens_v_full, ref_v, measured_idx, sigma_v, rho_v)

# (6) Write the updated full field back to CSV files.
for i, filename in enumerate(member_filenames):
    # For member i, extract the i-th column from the updated ensemble.
    updated_u = ens_u_full_updated[:, i]
    updated_v = ens_v_full_updated[:, i]
    
    export_data = {
        "Ux": updated_u,
        "Uy": updated_v,
        "CellID": IDs_full
    }
    df = pd.DataFrame(export_data)
    
    output_file = os.path.join(outp_directory, f"member{i+1}.csv")
    df.to_csv(output_file, index=False, float_format="%.6f")
    # print(f"Updated full field for member {i+1} written to {output_file}")
