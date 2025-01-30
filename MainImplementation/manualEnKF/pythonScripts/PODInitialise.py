import numpy as np
import scipy.io
import sys

# Number of random fields to generate
N_members = 2
# Alternatively, read from command line:
# N_members = int(sys.argv[1])

# Load MATLAB data
mat = scipy.io.loadmat('../POD/POD_Initialisation_data.mat')

# Unpack variables from the .mat file
Y_mean   = mat['Y_mean']      # Mean flow (size: [N, 1] or [N,])
a_mean   = mat['a_mean']      # Mean of time coeffs (size: [M, 1] or [1, M])
a_stddev = mat['a_stddev']    # Std dev of time coeffs (size: [M, 1] or [1, M])
phi      = mat['phi_crop']    # Truncated mode matrix (size: [N, M])

# For clarity, let's get dimensions:
#   N = number of spatial points/DOFs (rows of phi)
#   M = number of retained modes (columns of phi)
N, M = phi.shape

# We'll store each random field in a list
random_fields = []

for i in range(N_members):
    # Generate a random coefficient vector for each mode
    # a_mean and a_stddev might be 1D or 2D in shape, so we ensure they broadcast properly.

    # If a_mean / a_stddev have shape (M,) or (M,1), then the following works:
    a_rand = a_mean + a_stddev * np.random.randn(M, 1)  
    # If a_mean is (1,M), you might need .T shapes or to reshape accordingly. 
    # Adjust as needed if you see shape mismatches.

    # Reconstruct random flow field: Y_rand = Y_mean + phi * a_rand
    # Make sure Y_mean is also shape (N,1) or broadcastable:
    Y_rand = Y_mean + phi @ a_rand

    # Store or process Y_rand
    random_fields.append(Y_rand)

    # (Optional) print or inspect shapes
    print(f"Member {i+1}: Y_rand shape = {Y_rand.shape}")

# Now 'random_fields' is a list of size N_members,
# each entry is a NumPy array of shape (N, 1) (or (N,)) containing the random flow.

# If desired, you can convert this list to a single 2D array
# E.g., shape: (N, N_members) by concatenation
random_fields_array = np.hstack(random_fields)

# Do something with random_fields_array, such as saving to disk or further analysis
# Example: save back to a .mat or a NumPy file
# scipy.io.savemat('random_fields.mat', {'random_fields': random_fields_array})
# or
# np.save('random_fields.npy', random_fields_array)
