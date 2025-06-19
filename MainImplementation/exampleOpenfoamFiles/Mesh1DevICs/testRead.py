import re
import numpy as np

def read_vector_data(filename):
    """
    Reads vector data from an OpenFOAM file with the specified format.
    
    Parameters:
    - filename: str
        The path to the file containing the vector data.
    
    Returns:
    - np.ndarray
        A numpy array of shape (N, 3), where N is the number of vectors,
        and each row corresponds to a vector (x, y, z).
    """
    with open(filename, 'r') as file:
        content = file.read()

    # Extract the internalField block
    internal_field_match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+\d+\s+\((.*?)\)\s*;', content, re.DOTALL)
    if not internal_field_match:
        raise ValueError("Could not find the internalField block in the file.")

    vector_data_block = internal_field_match.group(1)

    # Extract individual vectors from the block
    vector_pattern = r'\((-?\d*\.?\d+(e[-+]?\d+)?)\s+(-?\d*\.?\d+(e[-+]?\d+)?)\s+(-?\d*\.?\d+(e[-+]?\d+)?)\)'
    vectors = re.findall(vector_pattern, vector_data_block)

    # Convert vectors to a numpy array
    vector_array = np.array([[float(x[0]), float(x[2]),] for x in vectors])

    return vector_array

def read_scalar_data(filename):
    """
    Reads scalar data from an OpenFOAM file with the specified format.
    
    Parameters:
    - filename: str
        The path to the file containing the scalar data.
    
    Returns:
    - np.ndarray
        A numpy array containing the scalar values.
    """
    with open(filename, 'r') as file:
        content = file.read()

    # Extract the internalField block
    internal_field_match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+\d+\s+\((.*?)\)\s*;', content, re.DOTALL)
    if not internal_field_match:
        raise ValueError("Could not find the internalField block in the file.")

    scalar_data_block = internal_field_match.group(1)

    # Extract individual scalar values from the block
    scalar_pattern = r'(-?\d*\.?\d+(e[-+]?\d+)?)'  # Matches float or scientific notation
    scalars = re.findall(scalar_pattern, scalar_data_block)

    # Convert scalars to a numpy array
    scalar_array = np.array([float(x[0]) for x in scalars])

    return scalar_array

# Example usage
filename = "p"  # Replace with the actual file path
try:
    scalar_data = read_scalar_data(filename)
    print("Scalar Data Shape:", scalar_data.shape)
    print("Sample Data:\n", scalar_data[:10])  # Print first 10 scalar values
except ValueError as e:
    print("Error:", e)

# Example usage
filename = "U"  # Replace with the actual file path
try:
    vector_data = read_vector_data(filename)
    print("Vector Data Shape:", vector_data.shape)
    print("Sample Data:\n", vector_data[:5])  # Print first 5 vectors
except ValueError as e:
    print("Error:", e)

print(vector_data.shape)
