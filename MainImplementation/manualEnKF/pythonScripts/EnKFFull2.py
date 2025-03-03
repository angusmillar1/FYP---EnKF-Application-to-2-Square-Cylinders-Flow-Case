import pandas as pd
import numpy as np
import os

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

def enkf_update_full_field_combined(ens, y, measured_idx, R, infl):
    """
    Perform an EnKF update for the full combined state vector.
    
    Parameters
    ----------
    ens : ndarray, shape (2N, N_e)
        Combined forecast ensemble (stacked u on top of v).
        Each column is one ensemble member.
    y : ndarray, shape (2P,)
        Combined observation vector (stacked u observations then v observations).
    measured_idx : ndarray, shape (2P,)
        Indices into the combined state that are measured.
    R : ndarray, shape (2P, 2P)
        The combined measurement noise covariance matrix.
    
    Returns
    -------
    ens_a : ndarray, shape (2N, N_e)
        The analysis (updated) ensemble for the full combined state.
    """
    N_full, N_e = ens.shape
    P = len(measured_idx)
    
    # (a) Generate measurement error perturbations in measurement space.
    eps = generate_perturbations_cholesky(R, N_e)  # shape (2P, N_e)
    
    # (b) Form the perturbed observations for each ensemble member.
    Y = np.tile(y, (N_e, 1)).T + eps   # shape (2P, N_e)
    
    # (c) Compute the full–state ensemble mean and anomalies.
    x_mean = np.mean(ens, axis=1)               # shape (2N,)
    X_ano  = (ens - x_mean[:, None]) * infl     # shape (2N, N_e)
    
    # (d) Extract the measurement anomalies using the measured indices.
    S = X_ano[measured_idx, :]           # shape (2P, N_e)
    
    # (e) Compute the innovation covariance in measurement space.
    cov_yy = (1.0/(N_e - 1)) * (S @ S.T) + R   # shape (2P, 2P)
    
    # (f) Compute the cross-covariance between full state and measurement space.
    P_xy = (1.0/(N_e - 1)) * (X_ano @ S.T)       # shape (2N, 2P)
    
    # (g) Compute the Kalman gain.
    K = P_xy @ np.linalg.inv(cov_yy)             # shape (2N, 2P)
    
    # (h) Get the forecasted measurement for each ensemble member.
    Hxf = ens[measured_idx, :]                   # shape (2P, N_e)
    
    # (i) EnKF update: add the innovation (Y - Hxf) to the forecast.
    ens_a = ens + K @ (Y - Hxf)                  # shape (2N, N_e)
    
    return ens_a

# ----------------------------
# Main code: reading data and updating the full field
# ----------------------------

# Directories:
redu_directory = "EnKFMeshData/reducedMeshData"
full_directory = "EnKFMeshData/fullMeshData"
outp_directory = "EnKFMeshData/filteredMeshData/"

# (1) Read the reference (measurement) data from the reduced mesh.
ref_filepath = os.path.join(redu_directory, "refSoln.csv")
ref_data = pd.read_csv(ref_filepath)
ref_u   = ref_data['Ux'].values
ref_v   = ref_data['Uy'].values
ref_IDs = ref_data['CellID'].values  
# (Assumption: ref_IDs gives the (0-indexed) locations in the full mesh corresponding to the measured cells)

# (2) Read the full–field ensemble from the full directory.
full_ens_u_list = []
full_ens_v_list = []
IDs_full = None

# Get all member files.
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
ens_u_full = np.column_stack(full_ens_u_list)  # shape: (N, Ne)
ens_v_full = np.column_stack(full_ens_v_list)    # shape: (N, Ne)
N_full = ens_u_full.shape[0]

# (3) Combine the u and v state vectors into one combined state.
# The combined state has shape (2N, Ne) where the first N rows are u and the next N are v.
ens_combined = np.vstack((ens_u_full, ens_v_full))

# (4) Combine the measurement vectors.
# Each measurement is taken at the same cell index for u and v.
# So the combined measurement vector is: [ref_u; ref_v]
y_combined = np.concatenate((ref_u, ref_v))
P = len(ref_IDs)  # number of measured points per field

# Build measured indices for the combined state.
# For u: use ref_IDs; for v: shift indices by N_full.
measured_idx_combined = np.concatenate((ref_IDs, ref_IDs + N_full))

# (5) Set measurement noise parameters and construct the combined noise covariance.
sigma_u, sigma_v = 0.05, 0.05    # standard deviations for Ux and Uy
rho_u, rho_v     = 0.0, 0.0       # assume no cross-point correlation
inflationFactor  = 1.00           # inflation applied to anomaly matrix to prevent collapse

R_u = generate_R(P, sigma_u, rho_u)
R_v = generate_R(P, sigma_v, rho_v)
# Create a block-diagonal covariance matrix for the combined measurement vector.
R_combined = np.block([
    [R_u,             np.zeros((P, P))],
    [np.zeros((P, P)), R_v]
])  # Need to change if rho becomes non-zero and there are quanitifyable correlations

# (6) Perform the combined EnKF update.
ens_combined_updated = enkf_update_full_field_combined(ens_combined, y_combined, measured_idx_combined, R_combined, inflationFactor)

# (7) Split the updated combined state back into u and v components.
ens_u_full_updated = ens_combined_updated[:N_full, :]
ens_v_full_updated = ens_combined_updated[N_full:, :]

# (8) Write the updated full field back to CSV files.
for i, filename in enumerate(member_filenames):
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
