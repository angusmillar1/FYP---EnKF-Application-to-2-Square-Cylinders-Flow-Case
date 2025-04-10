import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import time as timepkg
import os
import re
import numpy as np
import sys
from scipy.stats import linregress

# CHOOSE WHAT TO PLOT
wholeFieldOn = 1
probePlotOn = 1
plotAvgVar = 1
plotAssimInt = 1
printProgress = 1

if len(sys.argv) > 1 and sys.argv[1]:
    # Inherited inputs if calling from Allrun.py 
    num_members = int(sys.argv[1])  # Get number of members from parent script for spread plot
    assimInt = float(sys.argv[2])   # Get assimilation interval for plotting of vert lines
    # Plot everything when run in big sim
    wholeFieldOn = 1
    probePlotOn = 1
    plotAvgVar = 1
    plotAssimInt = 1
    printProgress = 0
else:
    # Equivalent inherited inputs if running independtly
    num_members = 5    # Manually set number of members for spread plot
    assimInt = 25       # Manually set assimilation interval for plotting of vert lines

# Other plotting inputs
probeNum = [0,2,16] # Choose probe points to plot for

if len(sys.argv) > 1 and sys.argv[1]:
    timeWindow = []     # Automatically select whole domain when run from allrun
else:
    timeWindow = [] # Manually select region in time to plot, eg could be [2, 5] or left empty for whole domain.




# File path
input_path = "outputs/"
output_path = "outputs/errorPlots/"



# ADD INDICES TO SAMPLE POINTS LOCATION FILE
probe_coords_file = input_path+"sample_points_locations.csv"
df = pd.read_csv(probe_coords_file)
if 'p' not in df.columns:
    df.insert(0, 'p', range(0, len(df)))    # Insert a new column 'p' with index starting from 1
    df.to_csv(probe_coords_file, index=False)
    if printProgress: print(f"Modified file saved as {probe_coords_file}")
else: printProgress: print(f"{probe_coords_file} already contains point indices")



# WHOLE FIELD ERRORS

if wholeFieldOn:
    if printProgress: print("Starting whole field error plots")
    # Read the CSV file
    try:
        # Load the data into a DataFrame
        data = pd.read_csv(input_path+"error_write.csv")

        # Filter the data if timeWindow is provided
        if timeWindow and len(timeWindow) == 2:
            t_min, t_max = timeWindow
            data = data[(data["T"] >= t_min) & (data["T"] <= t_max)]

        # Check if the required columns exist
        required_columns = ["T", "L1_u", "L1_v", "L1_tot", "L2_u", "L2_v", "L2_tot", "MSE_u", "MSE_v", "MSE_tot"]

        if all(col in data.columns for col in required_columns):
            # ---------------------------
            # L1 Norm Plot
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(data["T"], data["L1_u"], label="L1_u", marker='o')
            plt.plot(data["T"], data["L1_v"], label="L1_v", marker='s')
            # If you wish to include pressure later, uncomment the following:
            # plt.plot(data["T"], data["L1_p"], label="L1_p", marker='^')
            plt.plot(data["T"], data["L1_tot"], label="L1_tot", marker='d')

            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel("Time (T)", fontsize=12)
            plt.ylabel("L1 Norm", fontsize=12)
            plt.title("L1 Norm Over Time", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True)
            plt.tight_layout()

            # Save the L1 plot
            if not timeWindow: output_plot_path_L1 = output_path+"L1_plot.png"
            else: output_plot_path_L1 = output_path+f"L1_plot_windowed_{t_min}_{t_max}.png"
            plt.savefig(output_plot_path_L1, dpi=300)
            if printProgress: print(f"L1 plot saved as '{output_plot_path_L1}'")

            # Display the L1 plot for 5 seconds
            plt.show(block=False)
            start_plot_timer = timepkg.time()
            while timepkg.time() - start_plot_timer < 2:
                plt.pause(0.1)
            plt.close()

            # ---------------------------
            # L2 Norm Plot
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(data["T"], data["L2_u"], label="L2_u", marker='o')
            plt.plot(data["T"], data["L2_v"], label="L2_v", marker='s')
            # If you wish to include pressure later, uncomment the following:
            # plt.plot(data["T"], data["L2_p"], label="L2_p", marker='^')
            plt.plot(data["T"], data["L2_tot"], label="L2_tot", marker='d')

            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel("Time (T)", fontsize=12)
            plt.ylabel("L2 Norm", fontsize=12)
            plt.title("L2 Norm Over Time", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True)
            plt.tight_layout()

            # Save the L2 plot
            if not timeWindow: output_plot_path_L2 = output_path+"L2_plot.png"
            else: output_plot_path_L2 = output_path+f"L2_plot_windowed_{t_min}_{t_max}.png"
            plt.savefig(output_plot_path_L2, dpi=300)
            if printProgress: print(f"L2 plot saved as '{output_plot_path_L2}'")

            # Display the L2 plot for 5 seconds
            plt.show(block=False)
            start_plot_timer = timepkg.time()
            while timepkg.time() - start_plot_timer < 2:
                plt.pause(0.1)
            plt.close()

            # ---------------------------
            # MSE Plot
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(data["T"], data["MSE_u"], label="MSE_u", marker='o')
            plt.plot(data["T"], data["MSE_v"], label="MSE_v", marker='s')
            plt.plot(data["T"], data["MSE_tot"], label="MSE_tot", marker='d')

            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel("Time (T)", fontsize=12)
            plt.ylabel("MSE", fontsize=12)
            plt.title("MSE Over Time", fontsize=14)
            plt.legend(loc="upper right", fontsize=10)
            plt.grid(True)
            plt.tight_layout()

            # Save the MSE plot
            if not timeWindow: output_plot_path_MSE = output_path+"MSE_plot.png"
            else: output_plot_path_MSE = output_path+f"MSE_plot_windowed_{t_min}_{t_max}.png"
            plt.savefig(output_plot_path_MSE, dpi=300)
            if printProgress: print(f"MSE plot saved as '{output_plot_path_MSE}'")

            # Display the MSE plot for 5 seconds
            plt.show(block=False)
            start_plot_timer = timepkg.time()
            while timepkg.time() - start_plot_timer < 2:
                plt.pause(0.1)
            plt.close()

        else:
            print(f"The file does not contain the required columns: {required_columns}")
    except FileNotFoundError:
        print(f"File '{input_path}error_write.csv' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# MEASUREMENT POINTS ERROR TRACKING

if probePlotOn:
    if printProgress: print("Starting probe plots")

    # Base directory where the member folders are stored.
    base_dir = "memberRunFiles"

    # Initialize a dictionary to hold the data for each member and the reference solution.
    data = {}

    # Function to extract data from a given directory structure
    def extract_data(member_dir):
        """Extracts velocity data from a given member directory."""
        time_dirs = sorted(os.listdir(member_dir), key=lambda x: float(x))
        
        times_list = []
        u_data_list = []
        v_data_list = []
        
        for t in time_dirs:
            file_path = os.path.join(member_dir, t, "U")
            if not os.path.isfile(file_path):
                continue
            
            with open(file_path, "r") as f:
                lines = f.readlines()
            
            data_lines = lines[11:]
            
            for line in data_lines:
                if not line.strip():
                    continue

                tokens = line.strip().split()
                try:
                    time_val = float(tokens[0])
                except ValueError:
                    continue
                times_list.append(time_val)
                
                probe_strings = re.findall(r'\((.*?)\)', line)
                u_row = []
                v_row = []
                for ps in probe_strings:
                    numbers = ps.split()
                    if len(numbers) < 2:
                        continue
                    try:
                        u_val = float(numbers[0])
                        v_val = float(numbers[1])
                    except ValueError:
                        continue
                    u_row.append(u_val)
                    v_row.append(v_val)
                
                u_data_list.append(u_row)
                v_data_list.append(v_row)
        
        if times_list:
            return {
                "time": np.array(times_list),
                "u": np.array(u_data_list),
                "v": np.array(v_data_list),
            }
        else:
            return {
                "time": np.array([]),
                "u": np.array([]),
                "v": np.array([]),
            }

    # Extract ensemble member data
    for member in range(1, num_members + 1):
        member_dir = os.path.join(base_dir, f"member{member}", "postProcessing", "velocityProbes")
        data[f"member{member}"] = extract_data(member_dir)
        if printProgress: print(f"Data read for member {member}/{num_members}")

    # Extract reference solution data
    ref_dir = os.path.join(base_dir, "refSoln", "postProcessing", "velocityProbes")
    data["refSoln"] = extract_data(ref_dir)
    if printProgress: print("Data read for reference solution")

    # Sort member keys
    members = sorted([k for k in data.keys() if "member" in k], key=lambda m: int(m.replace("member", "")))
    num_members = len(members)

    # Use the time array from the first member (assuming identical simulation times).
    time = data[members[0]]["time"]
    num_timesteps = len(time)
    num_probe_points = data[members[0]]["u"].shape[1]

    # Stack the data from each member into 3D arrays.
    u_ensemble = np.empty((num_members, num_timesteps, num_probe_points))
    v_ensemble = np.empty((num_members, num_timesteps, num_probe_points))

    for i, mem in enumerate(members):
        u_ensemble[i, :, :] = data[mem]["u"]
        v_ensemble[i, :, :] = data[mem]["v"]

    # Now, if timeWindow is set, filter the ensemble data and time arrays.
    if timeWindow and len(timeWindow) == 2:
        t_min, t_max = timeWindow
        mask = (time >= t_min) & (time <= t_max)

        # Filter ensemble times and velocities
        time = time[mask]
        u_ensemble = u_ensemble[:, mask, :]
        v_ensemble = v_ensemble[:, mask, :]

    # Compute the ensemble mean, max, and min across members (axis 0).
    u_mean = np.mean(u_ensemble, axis=0)
    u_max = np.max(u_ensemble, axis=0)
    u_min = np.min(u_ensemble, axis=0)

    v_mean = np.mean(v_ensemble, axis=0)
    v_max = np.max(v_ensemble, axis=0)
    v_min = np.min(v_ensemble, axis=0)

    if printProgress: print("Ensemble statistics found")

    # Extract the reference solution data
    time_ref = data["refSoln"]["time"]
    u_ref = data["refSoln"]["u"]
    v_ref = data["refSoln"]["v"]

    # If you want to *also* filter reference solution data by the same timeWindow:
    if timeWindow and len(timeWindow) == 2 and len(time_ref) > 0:
        ref_mask = (time_ref >= t_min) & (time_ref <= t_max)
        time_ref = time_ref[ref_mask]
        u_ref = u_ref[ref_mask, :]
        v_ref = v_ref[ref_mask, :]

    # Get the positions of the probe points for clearer analysis
    probe_coords = pd.read_csv(probe_coords_file, skiprows=1, header=None).values

    # --- Plotting the results ---
    if printProgress: print("Starting plotting")

    for p in probeNum:

        # Select the relevant coordinates
        index, x_p, y_p, z_p = probe_coords[p]

        # Plot for the u component.
        plt.figure(figsize=(12, 6))
        plt.fill_between(time, u_min[:, p], u_max[:, p], alpha=0.3, color='C0', label='Ensemble Spread')
        plt.plot(time, u_mean[:, p], label='Ensemble Mean', color='C0')
        if u_ref.shape[0] > 0:
            plt.plot(time_ref, u_ref[:, p], '--', color='C2', label='Reference Solution')

        plt.xlabel("Time")
        plt.ylabel("u velocity")
        plt.title(f"Ensemble u Velocity at Probe ({x_p:.2f}, {y_p:.2f})")
        plt.grid(True)
        plt.legend(loc="best")

        if plotAssimInt:
            # Use MultipleLocator for vertical lines every 0.1 time units
            ax = plt.gca()  # Get current axes
            ax.xaxis.set_major_locator(ticker.MultipleLocator(assimInt))
            ax.grid(which='major', axis='x', linestyle='--', color='black')    

        plt.tight_layout()
        if not timeWindow: plt.savefig(output_path+f"U_probe_series_point{p}.png", dpi=300)
        else: plt.savefig(output_path+f"U_probe_series_point{p}_windowed_{t_min}_{t_max}.png", dpi=300)

        # Show for a short period
        plt.show(block=False)
        start_plot_timer = timepkg.time()
        while timepkg.time() - start_plot_timer < 2:
            plt.pause(0.1)
        plt.close()

        # Plot for the v component.
        plt.figure(figsize=(12, 6))
        plt.fill_between(time, v_min[:, p], v_max[:, p], alpha=0.3, color='C1', label='Ensemble Spread')
        plt.plot(time, v_mean[:, p], label='Ensemble Mean', color='C1')
        if v_ref.shape[0] > 0:
            plt.plot(time_ref, v_ref[:, p], '--', color='C3', label='Reference Solution')

        plt.xlabel("Time")
        plt.ylabel("v velocity")
        plt.title(f"Ensemble v Velocity at Probe ({x_p:.2f}, {y_p:.2f})")
        plt.grid(True)
        plt.legend(loc="best")

        if plotAssimInt:
            # Use MultipleLocator for vertical lines every 0.1 time units
            ax = plt.gca()  # Get current axes
            ax.xaxis.set_major_locator(ticker.MultipleLocator(assimInt))
            ax.grid(which='major', axis='x', linestyle='--', color='black')    

        plt.tight_layout()
        if not timeWindow: plt.savefig(output_path+f"V_probe_series_point{p}.png", dpi=300)
        else: plt.savefig(output_path+f"V_probe_series_point{p}_windowed_{t_min}_{t_max}.png", dpi=300)

        # Show for a short period
        plt.show(block=False)
        start_plot_timer = timepkg.time()
        while timepkg.time() - start_plot_timer < 2:
            plt.pause(0.1)
        plt.close()


# SPREAD TRACKING AND QUANTIFICATION

def exponential_window_fit(x, y, assimInt, num_points=100, skip_first=False, fixed_lambda=None):
    """
    Perform exponential fits on windows of data that reset every 'assimInt'.

    The model is:
        y(t) = A * exp(lambda * t)
    which is linearized as:
        ln(y) = ln(A) + lambda * t

    If fixed_lambda is provided (not None), then lambda is set to that value and only A is estimated.
    
    Parameters:
        x : array-like
            The time (or independent variable) values.
        y : array-like
            The variance (or dependent variable) values. Must be positive.
        assimInt : float
            The assimilation interval that defines the length of each window.
        num_points : int, optional
            Number of points for the smooth fitted curve in each window.
        skip_first : bool, optional
            If True, skip the first data point in each window before fitting.
        fixed_lambda : float or None, optional
            If provided, fixes lambda to this value and only estimates A.
    
    Returns:
        fit_results : list of dicts
            Each dictionary contains:
              - 'window_start': start value of the window,
              - 'window_end': end value of the window,
              - 'x_fit': array of x values for the fitted curve,
              - 'y_fit': array of y values for the fitted curve,
              - 'A_fit': fitted A parameter,
              - 'lambda_fit': fitted lambda parameter (or the fixed value),
              - 'r_value': correlation coefficient of the fit (only for full regression).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    fit_results = []
    # Define window boundaries based on the assimilation interval
    window_starts = np.arange(x[0], x[-1], assimInt)
    
    for start in window_starts:

        # Select data in the current window: [start, start+assimInt)
        mask = (x >= start) & (x < start + assimInt)
        x_window = x[mask]
        y_window = y[mask]
        
        if len(x_window) < 2:
            continue  # Not enough data for a fit
        
        # Optionally skip the first data point in each window
        if skip_first and len(x_window) > 2:
            x_window = x_window[1:]
            y_window = y_window[1:]
        
        # Remove any non-positive y values (needed for logarithm)
        pos_mask = y_window > 0
        x_fit_data = x_window[pos_mask]
        y_fit_data = y_window[pos_mask]
        
        if len(x_fit_data) < 2:
            continue
        
        # If a fixed lambda is provided, only regress for A
        if (fixed_lambda is not None) and (start != 0.0):
            # Compute ln(A) as the average difference: ln(y) - fixed_lambda*x.
            ln_A_est = np.mean(np.log(y_fit_data) - fixed_lambda * x_fit_data)
            A_fit = np.exp(ln_A_est)
            lambda_fit = fixed_lambda
            r_value = None  # Not computed in this case
        else:
            # Full regression: estimate both ln(A) and lambda.
            log_y = np.log(y_fit_data)
            regression = linregress(x_fit_data, log_y)
            slope = regression.slope
            intercept = regression.intercept
            A_fit = np.exp(intercept)
            lambda_fit = slope
            r_value = regression.rvalue
        
        # Generate a smooth fitted curve for this window
        x_fit_curve = np.linspace(x_fit_data.min(), x_fit_data.max(), num_points)
        y_fit_curve = A_fit * np.exp(lambda_fit * x_fit_curve)
        
        fit_results.append({
            'window_start': start,
            'window_end': start + assimInt,
            'x_fit': x_fit_curve,
            'y_fit': y_fit_curve,
            'A_fit': A_fit,
            'lambda_fit': lambda_fit,
            'r_value': r_value
        })
    
    return fit_results

def collapse_rate_regression(fit_results):
    """
    Given a list of fit results from a windowed exponential fit,
    this function extracts the end-of-window fitted values and
    corresponding end times, and performs a regression assuming:
    
        y(t) = B * exp(-mu * t)
    
    Taking the log gives:
    
        log(y) = log(B) - mu * t.
    
    Parameters:
        fit_results: list of dicts, each containing at least:
            - 'window_end': the end time of the window.
            - 'y_fit': an array of y values from the fitted curve in that window.
    
    Returns:
        B: the estimated prefactor in the collapse model.
        mu: the estimated collapse rate.
        reg: the regression result from linregress.
        t_values: the array of window end times used.
        y_values: the corresponding fitted y values used.
    """
    t_values = []
    y_values = []
    
    for fit in fit_results:
        # Use the window's end time as the time coordinate,
        # and the last value of the fitted curve as the y value.
        t_values.append(fit['window_end'])
        y_values.append(fit['y_fit'][-1])
    
    t_values = np.array(t_values)
    y_values = np.array(y_values)
    
    # Remove any non-positive y values (required for log transform)
    mask = y_values > 0
    t_values = t_values[mask]
    y_values = y_values[mask]
    
    # Take logarithm and perform linear regression
    log_y = np.log(y_values)
    reg = linregress(t_values, log_y)
    
    mu = -reg.slope  # because log(y) = log(B) - mu*t
    B = np.exp(reg.intercept)
    
    return B, mu, reg, t_values, y_values

if plotAvgVar:
    if printProgress: print("Starting spread analysis")
    # --- Compute and plot the average variance for u and v separately ---

    # 1) Compute per-timestep variance over the ensemble (axis 0) for each probe point.
    #    u_var and v_var have shape (num_timesteps, num_probe_points)
    u_var = np.var(u_ensemble, axis=0)
    v_var = np.var(v_ensemble, axis=0)

    # 2) Average the variance over all probe points for each timestep.
    #    This results in one average variance value per timestep for u and v.
    average_u_variance = np.mean(u_var, axis=1)
    average_v_variance = np.mean(v_var, axis=1)
    combined_average_variance = (average_u_variance + average_v_variance) / 2

    # 3) Plot the average variances for u and v on the same figure for comparison.
    plt.figure(figsize=(10, 5))
    plt.plot(time, average_u_variance, label='Average u Variance', color='C0')
    plt.plot(time, average_v_variance, label='Average v Variance', color='C1')
    plt.plot(time, combined_average_variance, label='Combined Average Variance', color='C2', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Average Variance")
    plt.title("Ensemble Average Variance for u and v Components")
    plt.grid(True)
    plt.legend(loc="best")

    if plotAssimInt:
        # Use MultipleLocator for vertical lines every few time units
        ax = plt.gca()  # Get current axes
        ax.xaxis.set_major_locator(ticker.MultipleLocator(assimInt))
        ax.grid(which='major', axis='x', linestyle='--', color='black')  

    plt.tight_layout()

    if not timeWindow: output_plot_path_avgvar = output_path + "average_variance.png"
    else: output_plot_path_avgvar = output_path+f"average_variance_windowed_{t_min}_{t_max}.png"

    plt.savefig(output_plot_path_avgvar, dpi=300)
    plt.show(block=False)
    start_plot_timer = timepkg.time()
    while timepkg.time() - start_plot_timer < 2:
        plt.pause(0.1)
    plt.close()


    # Fit exponential curves to the data, as Lyapunov predicts, to allow for tracking of the collapse rate

    # Get the fits:
    fits = exponential_window_fit(time, combined_average_variance, assimInt, skip_first=True, fixed_lambda=0.2)

    B_est, mu_est, reg, t_vals, y_vals = collapse_rate_regression(fits)

    # Plot data and fits:
    plt.figure(figsize=(10, 6))
    plt.plot(time, combined_average_variance, 'o', label='Data', markersize=3)
    for fit in fits:
        plt.plot(fit['x_fit'], fit['y_fit'], '--')#, 
                 #label=f"Fit {fit['window_start']:.0f}-{fit['window_end']:.0f} (λ={fit['lambda_fit']:.3f})")
    
    # Create a smooth curve for the collapse rate regression: B*exp(-mu*t)
    t_reg = np.linspace(0, t_vals.max(), 200)
    y_reg = B_est * np.exp(-mu_est * t_reg)
    plt.plot(t_reg, y_reg, 'k-', linewidth=2, label=f"Collapse Fit (μ={mu_est:.3f})")
    # Highlight the pivot points (end-of-window values) used in the collapse regression
    plt.plot(t_vals, y_vals, 'ro', markersize=6, label="Pivot Points")

    plt.xlabel("Time")
    plt.ylabel("Variance")
    plt.title("Exponential Fits per Assimilation Window")
    plt.legend(loc="best", fontsize='small')
    if plotAssimInt:
        # Use MultipleLocator for vertical lines every few time units
        ax = plt.gca()  # Get current axes
        ax.xaxis.set_major_locator(ticker.MultipleLocator(assimInt))
        ax.grid(which='major', axis='x', linestyle='--', color='black')  
    plt.grid(True)

    if not timeWindow: output_plot_path_varfit = output_path + "average_variance_fit.png"
    else: output_plot_path_varfit = output_path+f"average_variance_fit_windowed_{t_min}_{t_max}.png"
    plt.savefig(output_plot_path_varfit, dpi=300)

    plt.show(block=False)
    start_plot_timer = timepkg.time()
    while timepkg.time() - start_plot_timer < 5:
        plt.pause(0.1)
    plt.close()




