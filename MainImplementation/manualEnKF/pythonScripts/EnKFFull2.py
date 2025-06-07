# Implements the EnKF update process
# Requires full processed fields from the ensemble, and measurement point data from the ensemble and ref soln
# Calculates and applies innovation and writes updated fields to csvs for later writing to openfoam format

import pandas as pd
import numpy as np
import os


# Helper functions

# Generate simple noise matrix from defined std dev and cross covariance constant 
def generate_R(N, sigma, rho):
    R = np.full((N, N), rho * sigma**2)
    np.fill_diagonal(R, sigma**2)
    return R

# Generates random measurement noise to add to measurements y based on stats of noise matrix R
def generate_perturbations_cholesky(R, P):
    # Generate P samples (columns) from N(0,R) using a Cholesky factorization
    # R must be symmetric positive definite
    N = R.shape[0]
    L = np.linalg.cholesky(R)
    epsilons = np.zeros((N, P))
    for i in range(P):
        z = np.random.randn(N)
        epsilons[:, i] = L @ z
    return epsilons

# Compute a weight matrix based on the Euclidean distance between state and measurement positions
def compute_localization_weights(state_positions, meas_positions, L):
    # Compute a pairwise distance matrix (n x m)
    dists = np.linalg.norm(state_positions[:, None, :] - meas_positions[None, :, :], axis=2)
    # Define weights decaying with distance from points
    W = np.exp(-0.5 * (dists / L)**2)
    return W

# Compute and apply innovations
def enkf_update_full_field_combined(ens, y, measured_idx, R, infl, W):
    
    # Get shape of ensemble
    N_full, N_e = ens.shape     # state dimension (2 * no. of cells), no. of samples (ensemble width)
    P = len(measured_idx)       # no. of measuremnt points
    
    # Generate measurement error perturbations in measurement space
    eps = generate_perturbations_cholesky(R, N_e) 
    
    # Form the perturbed observations for each ensemble member
    Y = np.tile(y, (N_e, 1)).T + eps  
    
    # Compute the full–state ensemble mean and anomalies
    x_mean = np.mean(ens, axis=1)         
    X_ano  = (ens - x_mean[:, None]) * infl  
    
    # Extract the measurement anomalies using the measured indices (apply 'H' matrix)
    S = X_ano[measured_idx, :]     
    
    # Compute the innovation covariance in measurement space
    cov_yy = (1.0/(N_e - 1)) * (S @ S.T) + R 
    
    # Compute the cross-covariance between full state and measurement space
    P_xy = (1.0/(N_e - 1)) * (X_ano @ S.T)   
    
    # Compute the Kalman gain.
    # Gain without localisation (only for comparison when using localisation)
    K_nonloc = P_xy @ np.linalg.inv(cov_yy)
    # Gain with localisation - used in update, set to unity externally if localisation not wanted
    K = (P_xy * W) @ np.linalg.inv(cov_yy)

    # Display comparison of gain magnitudes
    # print("Max gain non-localized:", np.abs(K_nonloc).max())
    # print("Max gain localized:", np.abs(K).max())

    # Get the forecasted measurement for each ensemble member
    Hxf = ens[measured_idx, :]      # In practice: extract ensemble data at measurement points
    
    # EnKF update: add the innovation (Y - Hxf) to the forecast.
    ens_a = ens + K @ (Y - Hxf)                  # shape (2N, N_e)
    
    return ens_a


# Main code: reading data and updating the full field

# Directories:
redu_directory = "EnKFMeshData/reducedMeshData"     # Data from ens and ref at meas pts
full_directory = "EnKFMeshData/fullMeshData"        # Data from full ens
outp_directory = "EnKFMeshData/filteredMeshData/"   # Updated fields for propagation
cellCentInpDir = "inputs/cellCentres"               # Mesh info for localisation weights

# Read the reference (measurement) data from the reduced mesh.
ref_filepath = os.path.join(redu_directory, "refSoln.csv")
ref_data = pd.read_csv(ref_filepath)
ref_u   = ref_data['Ux'].values
ref_v   = ref_data['Uy'].values
ref_IDs = ref_data['CellID'].values  
# Assumes ref_IDs gives the (0-indexed) locations in the full mesh corresponding to the measured cells
# This is the format written earlier in the Allrun process

# Read the full–field ensemble from the full directory
full_ens_u_list = []
full_ens_v_list = []
IDs_full = None

# Get all member files
member_filenames = sorted([f for f in os.listdir(full_directory) if f.startswith("member") and f.endswith(".csv")])
Ne = len(member_filenames)
for filename in member_filenames:
    filepath = os.path.join(full_directory, filename)
    data = pd.read_csv(filepath)
    # Assume each csv has columns "Ux", "Uy", "CellID" - std allrun format
    if IDs_full is None: IDs_full = data['CellID'].values
    full_ens_u_list.append(data['Ux'].values)
    full_ens_v_list.append(data['Uy'].values)

# Convert lists into ensemble arrays - each column corresponds to one ensemble member
ens_u_full = np.column_stack(full_ens_u_list)  
ens_v_full = np.column_stack(full_ens_v_list)   
N_full = ens_u_full.shape[0]

# Combine the u and v state vectors into one combined state of length 2*Ncells and width Ne
ens_combined = np.vstack((ens_u_full, ens_v_full))

# Combine the measurement vectors.
y_combined = np.concatenate((ref_u, ref_v))
P = len(ref_IDs)  # number of measured points per field

# Build measured indices for the combined state
measured_idx_combined = np.concatenate((ref_IDs, ref_IDs + N_full))

# Set measurement noise parameters and construct the combined noise covariance.
sigma_u, sigma_v = 0.05, 0.05       # standard deviations for Ux and Uy
rho_u, rho_v     = 0.0, 0.0         # 0 if assume no cross-point correlation
inflationFactor  = 1.00             # inflation linearly applied to anomaly matrix to prevent collapse
L = 2                               # covariance localisation characteristic length  

R_u = generate_R(P, sigma_u, rho_u)
R_v = generate_R(P, sigma_v, rho_v)
# Create a block-diagonal covariance matrix for the combined measurement vector.
R_combined = np.block([
    [R_u,             np.zeros((P, P))],
    [np.zeros((P, P)), R_v]
])  # Need to change if rho becomes non-zero and there are quanitifyable correlations

# Get full mesh cell centre information and calculate weights for covariance localisation
Cx = np.loadtxt(os.path.join(cellCentInpDir,"mesh2_Cx"))
Cy = np.loadtxt(os.path.join(cellCentInpDir,"mesh2_Cy"))
state_pos = np.column_stack((Cx, Cy))
meas_pos = state_pos[ref_IDs, :]

W = compute_localization_weights(state_pos, meas_pos, L)
W_combined = np.block([
    [W, W],
    [W, W]])
# print("W min, max, mean:", W.min(), W.max(), W.mean())

# Overwrite weightings with ones if no localisation is required
W_combined = np.ones((2*N_full, 2*P)) # Comment if localisation required

# Perform the combined EnKF update.
ens_combined_updated = enkf_update_full_field_combined(ens_combined, y_combined, measured_idx_combined, R_combined, inflationFactor, W_combined)

# Split the updated combined state back into u and v components.
ens_u_full_updated = ens_combined_updated[:N_full, :]
ens_v_full_updated = ens_combined_updated[N_full:, :]

# Write the updated full field back to CSV files.
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
