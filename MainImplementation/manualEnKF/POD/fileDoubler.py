# Define the function to duplicate the values
def duplicate_values(input_filename):
    
    # Open the input file and read the lines
    with open(input_filename, 'r') as infile:
        lines = infile.readlines()
    
    # Prepare the output filename
    output_filename = input_filename.replace('.txt', '_doubled.txt')
    
    # Open the output file for writing
    with open(output_filename, 'w') as outfile:
        for line in lines:
            value = line.strip()  # Remove any leading/trailing whitespace
            if value:  # Ensure the line is not empty
                # Write the value twice, each on a new line
                outfile.write(f"{value}\n{value}\n")
    
    print(f"Duplicated values have been written to {output_filename}")

# Run the function
input_filename = "cellVolumes_mesh1.txt"
duplicate_values(input_filename)
