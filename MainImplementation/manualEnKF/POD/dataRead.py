import os
import re
import numpy as np
import pandas as pd
import time as timepkg

def read_velocity_data(case_dir, start_time, end_time, output_csv):
    """
    Read OpenFOAM velocity files and write the extracted data to a CSV.

    Parameters:
        case_dir (str): Path to the OpenFOAM case directory.
        start_time (int): Start time (e.g., 0).
        end_time (int): End time (e.g., 500).
        output_csv (str): Name of the output CSV file.
    """
    snapshots = []  # To store all snapshots

    for time in range(start_time, end_time + 1):
        velocity_file = os.path.join(case_dir, str(time), "U")
        if not os.path.exists(velocity_file):
            print(f"File not found: {velocity_file}")
            continue

        # Read the velocity file
        with open(velocity_file, 'r') as f:
            lines = f.readlines()

        # Locate the start of internalField data
        try:
            start_index = next(
                i for i, line in enumerate(lines) if "internalField" in line and "nonuniform List<vector>" in line
            ) + 2  # Data starts two lines after this line
            num_entries = int(lines[start_index - 1].strip())  # The number of entries is on the line before the data
            # print(f"Time step {time}: Expected number of entries = {num_entries}")
            end_index = start_index + num_entries +1
        except (StopIteration, ValueError) as e:
            print(f"Failed to parse {velocity_file}. Error: {e}")
            continue
        
        # print(f"Start index {start_index}, end index {end_index}")
        # print(f"Time step {time}: Actual number of lines read = {len(lines[start_index:end_index])}")
        # if len(lines[start_index:end_index]) != num_entries:
        #     print(f"Mismatch: Expected {num_entries}, Found {len(lines[start_index:end_index])}")

        # match_count = 0
        # for line in lines[start_index:end_index]:
        #     match = re.match(r"\(\s*([-+]?\d*\.?\d+|0)\s+([-+]?\d*\.?\d+|0)\s+[-+]?\d*\.?\d+|0\s*\)", line.strip())
        #     if match:
        #         match_count += 1
        #     if not match:
        #         print(f"Skipped line: {repr(line.strip())}")
        # print(f"Time step {time}: Matched lines = {match_count}")
        # if match_count != num_entries:
        #     print(f"Mismatch: Expected {num_entries}, Found {match_count}")


        # Extract and process the velocity data
        snapshot = []
        for line in lines[start_index:end_index]:
            line = line.strip()
            # print(line)
            if line.startswith("(") and line.endswith(")"):
                # Remove parentheses and split by spaces
                components = line[1:-1].split()
                if len(components) == 3:  # Ensure three components are present
                    u = float(components[0])  # x-component
                    v = float(components[1])  # y-component
                    snapshot.append(u)
                    snapshot.append(v)
                else:
                    print(f"Malformed line: {line}")
            else:
                print(f"Skipped line: {line}")

        # Print the size of the snapshot for the current time step
        print(f"Time step {time}: snapshot size = {len(snapshot)}")
        # timepkg.sleep(0.5)
        # Append this snapshot to the list
        snapshots.append(snapshot)

    # Print the total size of the snapshots matrix
    if snapshots:
        print(f"Total snapshots collected: {len(snapshots)}")
        print(f"Snapshot matrix dimensions: {len(snapshots[0])} rows, {len(snapshots)} columns")
        snapshots_matrix = np.array(snapshots).T  # Transpose so each column is a snapshot
    else:
        print("No data found in the provided files.")
        return

    # Write to CSV
    df = pd.DataFrame(snapshots_matrix)
    df.to_csv(output_csv, sep=';', index=False, header=False)
    print(f"Data successfully written to {output_csv}")


# Parameters
case_directory = "../../../OpenfoamTestRuns/Good/Square_Cylinders_Non_Linear_Mesh1Dvlpd"  # Replace with your OpenFOAM case directory
start_time = 0  # Start time step
end_time = 500  # End time step
output_filename = "velocity_data.csv"  # Output CSV file name

# Run the function
read_velocity_data(case_directory, start_time, end_time, output_filename)
