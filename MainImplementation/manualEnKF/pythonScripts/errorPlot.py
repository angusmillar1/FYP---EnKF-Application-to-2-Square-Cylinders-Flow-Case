import pandas as pd
import matplotlib.pyplot as plt
import time

# File path
file_path = "outputs/error_write.csv"
output_plot_path = "outputs/error_plot.png"

# Read the CSV file
try:
    # Load the data into a DataFrame
    data = pd.read_csv(file_path)

    # Check if the required columns exist
    required_columns = ["T", "MSE_u", "MSE_v", "MSE_p", "MSE_tot"]
    if all(col in data.columns for col in required_columns):
        # Plot the data
        plt.figure(figsize=(10,6))
        plt.plot(data["T"], data["MSE_u"], label="MSE_u", marker='o')
        plt.plot(data["T"], data["MSE_v"], label="MSE_v", marker='s')
        plt.plot(data["T"], data["MSE_p"], label="MSE_p", marker='^')
        plt.plot(data["T"], data["MSE_tot"], label="MSE_tot", marker='d')

        # Add plot labels and legend
        plt.xlabel("Time (T)", fontsize=12)
        plt.ylabel("Error", fontsize=12)
        plt.title("Error Metrics Over Time", fontsize=14)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_plot_path, dpi=300)
        print(f"Plot of error metrics saved to '{output_plot_path}'")

        # Display the plot (for 5s)
        plt.show(block=False)
        start_plot_timer = time.time()
        while time.time() - start_plot_timer < 5:
            plt.pause(0.1)  # Allow GUI updates during the sleep
        plt.close()
    else:
        print(f"The file does not contain the required columns: {required_columns}")
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
