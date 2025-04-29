import os
import sys
import re
import glob
import gc
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ------------------------------------------------------
# Options and settings
# ------------------------------------------------------
# Grid and temporal resolution
N_x = 14  
N_y = 6  
N_z = 4  
timestep_interval = 5e-6

# Define the x-offset (in meters) from which to begin sampling.
x_offset = 0.05

# Steady state configuration
tol = 0.0075        # tolerance for steady state (relative change)
ignore_time = 2.0  # ignore first 2 seconds (20 data points)

# -------------------------
# Helper function: Determine particle distribution from folder (or filename)
# -------------------------
def get_particle_distribution(name):
    name_lower = name.lower()
    if "r0_r3" in name_lower:
        return "Binary A"
    elif "r0_r6" in name_lower:
        return "Binary B"
    elif "r0_r9" in name_lower:
        return "Binary C"
    elif "mcc" in name_lower:
        return "Poly-dispersed"
    else:
        return "Poly-dispersed"

# -------------------------
# Data processing functions
# -------------------------
def parse_simulation_folder(folder_name):
    mcc_pattern = r"sim_MCC_(?P<screw_spacing>\d+PD)_(?P<rotations>(?:[0-9.]+)?N)_(?P<rpm>\d+)"
    binary_pattern = r"sim_r0_r(?P<rAaa>[369])_(?P<screw_spacing>\d+PD)_(?P<rotations>(?:[0-9.]+)?N)_(?P<rpm>\d+)"
    mcc_match = re.match(mcc_pattern, folder_name)
    if mcc_match:
        params = mcc_match.groupdict()
        params["simtype"] = "MCC"
        return params
    binary_match = re.match(binary_pattern, folder_name)
    if binary_match:
        params = binary_match.groupdict()
        params["simtype"] = "Binary"
        return params
    return None

def get_all_particle_files(sim_folder):
    file_pattern = os.path.join(sim_folder, "particles_*.vtk")
    files = glob.glob(file_pattern)
    files = [f for f in files if "boundingBox" not in os.path.basename(f)]
    if not files:
        return []
    def get_step(fname):
        base = os.path.basename(fname)
        m = re.search(r"particles_(\d+)\.vtk", base)
        return int(m.group(1)) if m else 0
    files.sort(key=get_step)
    return files

def load_vtk_data(filepath):
    try:
        mesh = pv.read(filepath)
        points = mesh.points
        df = pd.DataFrame(points, columns=["x", "y", "z"])
        for key, arr in mesh.point_data.items():
            if key.lower() == "v" and arr.ndim == 2 and arr.shape[1] == 3:
                df["vx"] = arr[:, 0]
                df["vy"] = arr[:, 1]
                df["vz"] = arr[:, 2]
            else:
                if arr.ndim > 1:
                    df[key] = arr[:, 0]
                else:
                    df[key] = arr
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}", flush=True)
        return None

def compute_mixing_metric(H, df, x_edges, y_edges, z_edges):
    total_particles = len(df)
    if total_particles == 0:
        return np.nan
    unique_radii, global_counts = np.unique(df["radius"], return_counts=True)
    x_global = global_counts / total_particles
    S_mix = np.sum(x_global * np.log(x_global))
    S = 0.0
    n_x, n_y, n_z = len(x_edges)-1, len(y_edges)-1, len(z_edges)-1
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                mask = ((df["x"] >= x_edges[i]) & (df["x"] < x_edges[i+1]) &
                        (df["y"] >= y_edges[j]) & (df["y"] < y_edges[j+1]) &
                        (df["z"] >= z_edges[k]) & (df["z"] < z_edges[k+1]))
                cell_df = df[mask]
                cell_count = len(cell_df)
                if cell_count == 0:
                    continue
                n_k = cell_count / total_particles
                unique_cell, cell_counts = np.unique(cell_df["radius"], return_counts=True)
                x_cell = cell_counts / cell_count
                cell_entropy = np.sum(x_cell * np.log(x_cell))
                S += n_k * cell_entropy
    return S / S_mix if S_mix != 0 else np.nan

# -------------------------
# Steady state functions
# -------------------------
def exponential_func(t, a, b, c):
    """
    Exponential function that increases at a decaying rate: f(t) = a * (1 - e^(-b*t)) + c
    where:
    a: amplitude of the exponential term
    b: rate of increase
    c: vertical offset
    """
    return a * (1 - np.exp(-b * t)) + c

def compute_exponential_fit(times, metrics):
    """ 
    Fits an exponential function to the mixing metric time series.
    Returns the fitted values and the fitted function parameters.
    """
    try:
        # Normalize times to start from 0 to help with numerical stability
        times_normalized = times - times[0]
        
        # Calculate initial guesses for parameters [a, b, c]
        # a: amplitude (max value - min value)
        # b: rate (inverse of characteristic time)
        # c: offset (minimum value)
        min_val = np.min(metrics)
        max_val = np.max(metrics)
        amplitude = max_val - min_val
        
        # Estimate characteristic time as time to reach 63% of amplitude
        target_val = min_val + 0.63 * amplitude
        idx = np.where(metrics >= target_val)[0]
        if len(idx) > 0:
            char_time = times_normalized[idx[0]]
            rate_guess = 1.0 / char_time if char_time > 0 else 1.0
        else:
            rate_guess = 1.0
            
        p0 = [amplitude, rate_guess, min_val]
        
        # Set parameter bounds
        # a: must be positive
        # b: must be positive
        # c: must be between 0 and 1 (mixing metric range)
        bounds = ([0, 0, 0], [1, np.inf, 1])
        
        # Fit the exponential function
        popt, _ = curve_fit(exponential_func, times_normalized, metrics, 
                           p0=p0, bounds=bounds, maxfev=10000)
        
        # Generate fitted values
        fitted = exponential_func(times_normalized, *popt)
        return fitted, popt
    except RuntimeError as e:
        print(f"Warning: Could not fit exponential function: {e}", flush=True)
        return metrics, None

def check_steady_state_exponential(times, fitted_metrics, tol=0.0075, window_steady=5, popt=None):
    """
    Checks if the fitted exponential metric has stabilized over a sliding window.
    If |fitted[i+window_steady-1] - fitted[i]| < tol * |fitted[i]|,
    steady state is assumed.
    """
    if len(fitted_metrics) < window_steady:
        return False, None, None, None, None

    # Calculate dynamic window size if not provided
    if window_steady == 5:  # Default value
        window_steady = calculate_dynamic_window(times, fitted_metrics, popt)

    for i in range(0, len(fitted_metrics) - window_steady + 1):
        start_time = times[i]
        end_time = times[i + window_steady - 1]
        start_val = fitted_metrics[i]
        end_val = fitted_metrics[i + window_steady - 1]
        if abs(end_val - start_val) < tol * abs(start_val):
            return True, end_time, start_time, start_val, end_val
    return False, None, None, None, None

def calculate_dynamic_window(times, metrics, popt=None):
    """
    Dynamically calculates an appropriate window size for steady state detection
    based on the total simulation time and the characteristic time of the exponential fit.
    
    Parameters:
    -----------
    times : array-like
        Time points in the simulation
    metrics : array-like
        Metric values
    popt : tuple, optional
        Parameters of the exponential fit [a, b, c]
        
    Returns:
    --------
    int
        Calculated window size
    """
    # Get the total simulation time
    total_time = times[-1] - times[0]
    
    # Calculate the number of data points
    n_points = len(times)
    
    # If we have the exponential fit parameters, use the characteristic time
    if popt is not None:
        # The characteristic time is 1/b for an exponential
        char_time = 1.0 / popt[1] if popt[1] > 0 else total_time / 10
        
        # Calculate window size based on characteristic time
        # Use a window that covers about 30% of the characteristic time (increased from 20%)
        # but ensure it's not too small or too large
        time_based_window = int(0.3 * char_time / (total_time / n_points))
        
        # Ensure window is at least 10 points (increased from 5) and at most 40% of the data (increased from 30%)
        min_window = 10
        max_window = int(0.4 * n_points)
        
        return max(min_window, min(time_based_window, max_window))
    
    # If no fit parameters, use a heuristic based on total time
    # Use a window that's about 20% of the total data points (increased from 10%)
    # but ensure it's not too small or too large
    heuristic_window = int(0.2 * n_points)
    
    # Ensure window is at least 10 points (increased from 5) and at most 40% of the data (increased from 30%)
    min_window = 10
    max_window = int(0.4 * n_points)
    
    return max(min_window, min(heuristic_window, max_window))

def plot_mixing_metric_with_exponential(full_times, full_metrics, filtered_times, fitted, 
                                   steady, steady_time, start_time, start_val, end_val, tol, window_steady, sim_name, output_dir):
    """
    Plots the raw mixing metric data over the full time range and overlays the fitted exponential.
    If steady state is reached, marks the steady state window.
    Otherwise, an annotation indicates that steady state was not reached.
    """
    plt.figure(figsize=(10, 7))
    
    # Plot raw data
    plt.plot(full_times, full_metrics, linestyle='-', color='b', label="Mixing Metric M")
    
    # Overlay the fitted exponential
    plt.plot(filtered_times, fitted, linestyle='--', color='orange', label="Fitted Exponential")
    
    # Plot the filtered data region
    plt.axvspan(filtered_times[0], filtered_times[-1], color='gray', alpha=0.1, label="Fitted Region")
    
    if steady:
        # Mark the steady state window
        plt.axvspan(start_time, steady_time, color='green', alpha=0.2, label="Steady State Window")
        plt.plot(steady_time, end_val, 'mo', markersize=8, label=f"Steady State: {end_val:.3f} at {steady_time:.2f}s")
        
        # Add tolerance and window information
        plt.text(0.5, 0.05, f"Tolerance: {tol*100:.1f}%\nWindow Size: {window_steady}", 
                 transform=plt.gca().transAxes, fontsize=10, color='black',
                 bbox=dict(facecolor='white', alpha=0.8))
    else:
        plt.text(0.5, 0.9, f"Steady state not reached\n(tol = {tol*100:.1f}%, window = {window_steady})",
                 transform=plt.gca().transAxes, fontsize=10, color='red',
                 bbox=dict(facecolor='white', alpha=0.8))
        # Add tolerance information even when steady state is not reached
        plt.text(0.5, 0.05, f"Tolerance: {tol*100:.1f}%\nWindow Size: {window_steady}", 
                 transform=plt.gca().transAxes, fontsize=10, color='black',
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel("Time (s)")
    plt.ylabel("Mixing Metric M")
    plt.title(f"Mixing Metric vs Time for {sim_name}")
    plt.ylim(0, 1)  # Fixed y-axis range
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(output_dir, f"{sim_name}_steady_state_exponential.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved steady state plot for {sim_name} to {plot_filename}")

# ------------------------------------------------------
# Global parameters and directories
# ------------------------------------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))
temp_temp_parent_dir = os.path.dirname(script_dir)
temp_parent_dir = os.path.dirname(temp_temp_parent_dir)
parent_dir = os.path.dirname(temp_parent_dir)

# Search in Outputs/Run2
outputs_folder = os.path.join(parent_dir, "Outputs", "Run2")

save_csv_location = os.path.join(temp_parent_dir, "Error (CSV)", "Standard Deviation (CSV)")

os.makedirs(save_csv_location, exist_ok=True)

barrel_length = 0.18
screw_radius = 0.00916
barrel_clearance = 0.00025
outer_radius = screw_radius + barrel_clearance

axis_separation_mm = {
    "1PD": 15.00347,
    "2PD": 15.56194,
    "3PD": 16.12041,
    "4PD": 16.67888,
    "5PD": 17.23735
}

# Update the x-direction grid so that it starts at x_offset instead of 0.
x_min = x_offset
x_max = barrel_length
x_edges = np.linspace(x_min, x_max, N_x + 1)
# For y_edges, use a fixed width based on the barrel geometry (using "1PD" as reference)
y_edges = np.linspace(-((axis_separation_mm["1PD"]/1000.0 + 2*outer_radius)/2.0),
                      ((axis_separation_mm["1PD"]/1000.0 + 2*outer_radius)/2.0), N_y + 1)
z_edges = np.linspace(-outer_radius, outer_radius, N_z + 1)

all_simulation_data = []

# -------------------------
# Main processing loop
# -------------------------
for folder in os.listdir(outputs_folder):
    folder_path = os.path.join(outputs_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    sim_params = parse_simulation_folder(folder)
    if sim_params is None:
        print(f"Skipping folder '{folder}': simulation parameters not recognized.", flush=True)
        continue

    screw_spacing = sim_params["screw_spacing"]
    if screw_spacing not in axis_separation_mm:
        print(f"Skipping folder '{folder}': Unrecognized screw spacing {screw_spacing}.", flush=True)
        continue

    axis_sep_m = axis_separation_mm[screw_spacing] / 1000.0
    y_half = (axis_sep_m + 2 * outer_radius) / 2.0
    y_min, y_max = -y_half, y_half
    z_min, z_max = -outer_radius, outer_radius
    y_edges = np.linspace(y_min, y_max, N_y + 1)
    z_edges = np.linspace(z_min, z_max, N_z + 1)

    particle_files = get_all_particle_files(folder_path)
    if not particle_files:
        print(f"No particle files found in folder '{folder}'. Skipping simulation.", flush=True)
        continue

    print(f"Processing simulation: {folder} ({len(particle_files)} timesteps)", flush=True)
    
    # Process particle files
    simulation_time_series = []
    for vtk_file in particle_files:
        base = os.path.basename(vtk_file)
        m = re.search(r"particles_(\d+)\.vtk", base)
        step = int(m.group(1)) if m else 0
        time = step * timestep_interval
        print(f"Processing step {step} from file: {vtk_file}", flush=True)
        df = load_vtk_data(vtk_file)
        if df is None or "radius" not in df.columns:
            print(f"Skipping file {vtk_file} due to missing data.", flush=True)
            continue
        # Only consider particles with x >= x_offset.
        df = df[df["x"] >= x_offset]
        coords = np.vstack([df["x"], df["y"], df["z"]]).T
        H, _ = np.histogramdd(coords, bins=[x_edges, y_edges, z_edges])
        mixing_val = compute_mixing_metric(H, df, x_edges, y_edges, z_edges)
        simulation_time_series.append({
            "time": time,
            "histogram": H,
            "particle_count": np.sum(H),
            "mixing_metric": mixing_val,
            "df": df,
        })
        # Clear memory after each file
        del df, coords, H, mixing_val           
        gc.collect()
    simulation_time_series.sort(key=lambda entry: entry["time"])

    # Process the current simulation
    times = [entry["time"] for entry in simulation_time_series]
    mix_series = [entry["mixing_metric"] for entry in simulation_time_series]
    
    # Calculate steady state information ONCE for this simulation
    steady_state_info = {}
    mask = np.array(times) >= ignore_time
    if np.any(mask):
        filtered_times = np.array(times)[mask]
        filtered_metrics = np.array(mix_series)[mask]
        fitted, popt = compute_exponential_fit(filtered_times, filtered_metrics)
        steady, steady_time, start_time, start_val, end_val = check_steady_state_exponential(filtered_times, fitted, tol=tol, window_steady=calculate_dynamic_window(filtered_times, filtered_metrics, popt=popt))
        
        # Store steady state information
        steady_state_info = {
            "steady": steady,
            "steady_time": steady_time,
            "start_time": start_time,
            "start_val": start_val,
            "end_val": end_val,
            "fitted": fitted,
            "filtered_times": filtered_times,
            "popt": popt,
            "window_steady": calculate_dynamic_window(filtered_times, filtered_metrics, popt=popt)
        }
        
        # Create steady state plot
        steady_plot_dir = os.path.join(save_csv_location, "Steady State Plots")
        os.makedirs(steady_plot_dir, exist_ok=True)
        plot_mixing_metric_with_exponential(times, mix_series, filtered_times, fitted, 
                                       steady, steady_time, start_time, start_val, end_val, 
                                       tol, steady_state_info["window_steady"], 
                                       folder, steady_plot_dir)
    else:
        steady = False
        steady_time = None
        steady_state_info = {
            "steady": False,
            "steady_time": None,
            "start_time": None,
            "start_val": None,
            "end_val": None,
            "fitted": None,
            "filtered_times": None,
            "popt": None,
            "window_steady": None
        }
    
    # Store steady state information in entries
    for entry in simulation_time_series:
        entry["steady_state_reached"] = steady_state_info["steady"]
        entry["steady_state_time"] = steady_state_info["steady_time"]
    
    # Process current simulation data
    current_sim_data = []
    for entry in simulation_time_series:
        row = [folder, entry["time"], entry["particle_count"], entry["mixing_metric"], 
               steady_state_info["steady"], steady_state_info["steady_time"]]
        current_sim_data.append(row)
    
    # Append to global data
    all_simulation_data.extend(current_sim_data)
    
    # Clear memory
    del simulation_time_series
    gc.collect()

# Save results after processing all simulations
print("Saving results...", flush=True)

# Save combined simulation summary data
df_all = pd.DataFrame(all_simulation_data, columns=["Simulation", "Time (s)", "Particle Count", "Mixing Metric M", "Steady State Reached", "Steady State Time (s)"])
combined_csv_filename = os.path.join(save_csv_location, "standard_deviation_run2.csv")
df_all.to_csv(combined_csv_filename, index=False)
print(f"Saved mixing metric timeseries with steady state information to {combined_csv_filename}", flush=True)
