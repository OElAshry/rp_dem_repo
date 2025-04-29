import os
import sys
import re
import glob
import pickle
import gc
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ------------------------------------------------------
# Options and settings
# ------------------------------------------------------
use_saved_data = False 
calculate_sensitivity = False   # if False, no sensitivity analysis is performed (only default scale used)
sensitivity_interval = 1       # Compute sensitivity every x timesteps

# Grid and temporal resolution
N_x = 14  
N_y = 6  
N_z = 4  
timestep_interval = 5e-6

# Only calculate sensitivity for these scale factors (if enabled)
selected_scales = [0.5, 2.0]
sensitivity_scale_labels = [f"M_scale_{round(s,2)}" for s in selected_scales]

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
# Utility plotting functions
# -------------------------
def plot_3d_grid(x_edges, y_edges, z_edges, stl_file):
    """
    Plots the 3D grid as a wireframe using PyVista and overlays an STL file.
    Displays the plot interactively.
    """
    plotter = pv.Plotter()
    grid = pv.MultiBlock()
    for x in x_edges:
        for y in y_edges:
            grid.append(pv.Line([x, y, z_edges[0]], [x, y, z_edges[-1]]))
    for x in x_edges:
        for z in z_edges:
            grid.append(pv.Line([x, y_edges[0], z], [x, y_edges[-1], z]))
    for y in y_edges:
        for z in z_edges:
            grid.append(pv.Line([x_edges[0], y, z], [x_edges[-1], y, z]))
    plotter.add_mesh(grid, color="blue", line_width=1)
    try:
        stl_mesh = pv.read(stl_file)
        plotter.add_mesh(stl_mesh, color="gray", opacity=0.5)
    except Exception as e:
        print(f"Error loading STL file: {e}", flush=True)
    plotter.add_axes(labels_off=False)
    plotter.show_grid()
    plotter.show()

def plot_outlet_heatmap(avg_outlet, y_edges, z_edges, save_path=None):
    if avg_outlet is None:
        print("No outlet data to plot.", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
    im = ax.imshow(avg_outlet.T, origin='lower',
                   extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],
                   aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Local Ratio")
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)

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

def compute_local_ratio_3d(df, x_edges, y_edges, z_edges):
    total_particles = len(df)
    n_x, n_y, n_z = len(x_edges)-1, len(y_edges)-1, len(z_edges)-1
    local_ratio = np.full((n_x, n_y, n_z), np.nan)
    if total_particles == 0:
        return local_ratio
    unique_radii, global_counts = np.unique(df["radius"], return_counts=True)
    x_global = global_counts / total_particles
    global_entropy = np.sum(x_global * np.log(x_global))
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
                unique_cell, cell_counts = np.unique(cell_df["radius"], return_counts=True)
                x_cell = cell_counts / cell_count
                cell_entropy = np.sum(x_cell * np.log(x_cell))
                local_ratio[i, j, k] = cell_entropy / global_entropy if global_entropy != 0 else np.nan
    return local_ratio

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

def average_outlet_heatmap(simulation_time_series, steady_time, x_index, y_edges, z_edges, ratio_key="local_ratio_3d"):
    slices = []
    for entry in simulation_time_series:
        if entry["time"] >= steady_time and ratio_key in entry:
            arr3d = entry[ratio_key]
            if arr3d is not None and arr3d.shape[0] > x_index:
                slices.append(arr3d[x_index, :, :])
    if len(slices) == 0:
        return None
    stack = np.stack(slices, axis=0)
    avg_outlet = np.nanmean(stack, axis=0)
    return avg_outlet

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

# Search in Outputs/Binary A
outputs_folder = os.path.join(parent_dir, "Outputs", "Binary A")

pickle_folder = os.path.join(temp_parent_dir, "Pickle Files")
save_csv_location = os.path.join(temp_parent_dir, "Segregation Analysis")
save_heatmaps_location = os.path.join(save_csv_location, "Outlet Mixing Heatmaps")

os.makedirs(save_csv_location, exist_ok=True)
os.makedirs(save_heatmaps_location, exist_ok=True)

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

outlet_x_index = N_x - 1

# -------------------------
# OPTIONAL: Plot the 3D grid to verify sample cells.
# -------------------------
# stl_file = os.path.join(parent_dir, "Inputs", "Meshes", "Barrel", "3PD_barrel.stl")
# plot_3d_grid(x_edges, y_edges, z_edges, stl_file)

all_simulation_data = []
all_mixing_metric_series = {}
simulation_results = {}  # Store results by folder
all_outlet_rows = []

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
    sim_data_file = os.path.join(pickle_folder, f"{folder}_simulation_data.pkl")
    
    # Load and process one simulation at a time
    if use_saved_data and os.path.exists(sim_data_file):
        print(f"Loading saved simulation data for {folder}", flush=True)
        with open(sim_data_file, "rb") as f:
            simulation_time_series = pd.read_pickle(f)
        print(f"Loaded saved simulation data for {folder}", flush=True)
    else:
        print(f"Creating new pickle file for {folder}", flush=True)
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
            local_ratio_3d = compute_local_ratio_3d(df, x_edges, y_edges, z_edges)
            simulation_time_series.append({
                "time": time,
                "histogram": H,
                "particle_count": np.sum(H),
                "mixing_metric": mixing_val,
                "df": df,
                "local_ratio_3d": local_ratio_3d
            })
            # Clear memory after each file
            del df, coords, H, mixing_val, local_ratio_3d            
            gc.collect()
        simulation_time_series.sort(key=lambda entry: entry["time"])
        with open(sim_data_file, "wb") as f:
            pickle.dump(simulation_time_series, f)
        print(f"Saved simulation data for {folder} to {sim_data_file}", flush=True)

    # Process the current simulation
    times = [entry["time"] for entry in simulation_time_series]
    mix_series = [entry["mixing_metric"] for entry in simulation_time_series]
    all_mixing_metric_series[folder] = (times, mix_series)
    
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
        row = [folder, entry["time"], entry["particle_count"], entry["mixing_metric"]]
        for scale in selected_scales:
            label = f"M_scale_{round(scale,2)}"
            row.append(None)
        row.extend([steady_state_info["steady"], steady_state_info["steady_time"]])
        current_sim_data.append(row)
    
    # Append to global data
    all_simulation_data.extend(current_sim_data)
    
    # Process outlet heatmap for current simulation
    if steady_state_info["steady_time"] is None:
        print(f"System has not reached steady state for simulation: {folder}", flush=True)
        steady_time = simulation_time_series[-1]["time"]
    else:
        steady_time = steady_state_info["steady_time"]
    
    avg_outlet = average_outlet_heatmap(simulation_time_series, steady_time, outlet_x_index, y_edges, z_edges, ratio_key="local_ratio_3d")
    if avg_outlet is not None:
        outlet_heatmap_path = os.path.join(save_heatmaps_location, f"{folder}_outlet_heatmap.png")
        plot_outlet_heatmap(avg_outlet, y_edges, z_edges, save_path=outlet_heatmap_path)
        
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
        Y, Z = np.meshgrid(y_centers, z_centers, indexing='ij')
        for i_y in range(Y.shape[0]):
            for i_z in range(Z.shape[1]):
                all_outlet_rows.append([folder, Y[i_y, i_z], Z[i_y, i_z], avg_outlet[i_y, i_z], steady_state_info["steady"], steady_state_info["steady_time"]])
    
    # Store simulation results for later analysis
    simulation_results[folder] = {
        "params": sim_params,
        "time_series": simulation_time_series,
        "steady_state_info": steady_state_info  # Store steady state info for reuse
    }
    
    # Clear memory
    del simulation_time_series
    del avg_outlet
    del Y, Z
    gc.collect()

# Save results after processing all simulations
print("Saving results...", flush=True)

# Save combined simulation summary data
base_columns = ["Simulation", "Time (s)", "Particle Count", "Mixing Metric M"]
all_columns = base_columns + sensitivity_scale_labels + ["Steady State Reached", "Steady State Time (s)"]
df_all = pd.DataFrame(all_simulation_data, columns=all_columns)
combined_csv_filename = os.path.join(save_csv_location, "all_simulation_data_plots_binaryA.csv")
df_all.to_csv(combined_csv_filename, index=False)
print(f"Saved combined simulation summary CSV", flush=True)

# Save outlet heatmap data
if all_outlet_rows:
    df_outlet_global = pd.DataFrame(all_outlet_rows, columns=["Simulation", "y (m)", "z (m)", "Local Ratio", "Steady State Reached", "Steady State Time (s)"])
    outlet_global_csv = os.path.join(save_heatmaps_location, "all_outlet_heatmap_data_plots_binaryA.csv")
    df_outlet_global.to_csv(outlet_global_csv, index=False)
    print(f"Saved global outlet heatmap data", flush=True)
else:
    print("No outlet heatmap data to save.", flush=True)

# -------------------------
# Analysis 1: Global Mixing Metric vs. Operating Conditions
# -------------------------
print("Processing global mixing analysis...", flush=True)

def compute_steady_state_metric(sim_time_series, steady_state_info):
    """
    Computes the steady state metric using pre-calculated steady state information.
    """
    if steady_state_info["steady"]:
        steady_time = steady_state_info["steady_time"]
        vals = [entry["mixing_metric"] for entry in sim_time_series if entry["time"] >= steady_time]
        return np.mean(vals) if vals else sim_time_series[-1]["mixing_metric"], True, steady_time
    else:
        return sim_time_series[-1]["mixing_metric"], False, sim_time_series[-1]["time"]

dist_rpm_groups = {}
for folder, result in simulation_results.items():
    pdist = get_particle_distribution(folder)
    rpm = result["params"]["rpm"]
    dist_rpm_groups.setdefault(pdist, {}).setdefault(rpm, []).append(folder)

global_mixing_rows = []
for pdist in sorted(dist_rpm_groups.keys()):
    for rpm in sorted(dist_rpm_groups[pdist].keys(), key=lambda x: int(x)):
        for folder in dist_rpm_groups[pdist].get(rpm, []):
            sim = simulation_results[folder]
            sim_params = sim["params"]
            spacing_val = int(re.search(r"(\d+)", sim_params["screw_spacing"]).group(1))
            screw_pitch = sim_params["rotations"].strip()
            match = re.search(r"([\d.]+)N", screw_pitch)
            pitch_val = float(match.group(1)) if match else 1.0
            steady_metric, reached_steady, steady_time = compute_steady_state_metric(sim["time_series"], sim["steady_state_info"])
            global_mixing_rows.append([folder, get_particle_distribution(folder), spacing_val, pitch_val, rpm, steady_metric, reached_steady, steady_time])
df_global_mixing = pd.DataFrame(global_mixing_rows, columns=["Simulation", "Distribution", "Screw Spacing (PD)", "Pitch (N)", "RPM", "Mixing Metric M", "Steady State Reached", "Steady State Time (s)"])
global_csv = os.path.join(save_csv_location, "global_mixing_data_plots_binaryA.csv")
df_global_mixing.to_csv(global_csv, index=False)
print(f"Saved global mixing data", flush=True)

# -------------------------
# Analysis 2: Axial Profile of Local Mixing (Segregation Analysis)
# -------------------------
def compute_axial_profile(sim_time_series, x_edges, steady_time):
    """
    Computes the average axial profile of local mixing (local ratio)
    over timesteps after steady state.
    """
    profiles = []
    for entry in sim_time_series:
        if entry["time"] >= steady_time and entry.get("local_ratio_3d") is not None:
            profile = np.nanmean(entry["local_ratio_3d"], axis=(1,2))
            profiles.append(profile)
    if len(profiles) == 0:
        return None
    avg_profile = np.nanmean(np.stack(profiles, axis=0), axis=0)
    return avg_profile

axial_data_rows = []
# Loop over each simulation in the results.
for folder, sim_data in simulation_results.items():
    sim = sim_data["time_series"]
    steady_state_info = sim_data["steady_state_info"]
    
    # Use the pre-calculated steady state information
    steady = steady_state_info["steady"]
    steady_time = steady_state_info["steady_time"]
    
    if not steady:
        # If steady state is not reached, use the last timestep as steady_time.
        steady_time = sim[-1]["time"]
        print(f"System has not reached steady state for simulation: {folder}. Using last timestep.", flush=True)
    
    # Compute the axial profile using the determined steady_time.
    axial_profile = compute_axial_profile(sim, x_edges, steady_time)
    if axial_profile is not None:
        x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2.0
        for pos, val in zip(x_midpoints, axial_profile):
            # Get particle distribution and spacing from simulation parameters.
            sim_params = simulation_results[folder]["params"]
            pdist = get_particle_distribution(folder)
            spacing_val = int(re.search(r"(\d+)", sim_params["screw_spacing"]).group(1))
            screw_pitch = sim_params["rotations"].strip()
            match = re.search(r"([\d.]+)N", screw_pitch)
            pitch_val = float(match.group(1)) if match else 1.0
            rpm_val = int(sim_params["rpm"])
            axial_data_rows.append([folder, pdist, spacing_val, pitch_val, rpm_val, pos, val, steady, steady_time])

df_axial = pd.DataFrame(axial_data_rows, 
    columns=["Simulation", "Distribution", "Screw Spacing (PD)", "Pitch (N)", "RPM", "Axial Position (m)", "Local Ratio", "Steady State Reached", "Steady State Time (s)"])
axial_csv = os.path.join(save_csv_location, "axial_profile_data_plots_binaryA.csv")
df_axial.to_csv(axial_csv, index=False)
print(f"Saved axial profile data to {axial_csv}", flush=True)
