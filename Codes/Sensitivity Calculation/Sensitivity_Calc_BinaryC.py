import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ------------------------------------------------------
# User Settings
# ------------------------------------------------------
# Set to True to analyze all simulations, or False to analyze only selected ones
analyze_all_simulations = False

# List of simulations to analyze (only used if analyze_all_simulations is False)
# Leave empty to analyze all simulations
selected_simulations = [
    "sim_r0_r9_1PD_0.5N_200",
    "sim_r0_r9_1PD_1.5N_1000",
    "sim_r0_r9_3PD_N_600",
    "sim_r0_r9_5PD_0.5N_200",
    "sim_r0_r9_5PD_1.5N_1000"
]

# ------------------------------------------------------
# Options
# ------------------------------------------------------
use_saved_data = False 
calculate_sensitivity = True   # if False, no sensitivity analysis is performed (only default scale used)
sensitivity_interval = 1       # Compute sensitivity every 20 timesteps (adjust as needed)

# Only calculate sensitivity for these scale factors (if enabled)
# MODIFY THIS LIST TO CHANGE THE SCALE FACTORS
selected_scales = [0.5, 1.0, 2]  # Add or remove scale factors as needed
sensitivity_scale_labels = [f"M_scale_{round(s,2)}" for s in selected_scales]
sensitivity_avg_particles_labels = [f"Avg_Particles_scale_{round(s,2)}" for s in selected_scales]

# Define the x-offset (in meters) from which to begin sampling.
x_offset = 0.05  # adjust as needed

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
        return "Unknown"

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

def compute_avg_particles_per_cell(H):
    """Calculate the average number of particles per non-empty cell."""
    # Flatten the 3D histogram
    flat_hist = H.flatten()
    # Count non-empty cells
    non_empty_cells = np.sum(flat_hist > 0)
    if non_empty_cells == 0:
        return 0
    # Calculate average particles per non-empty cell
    total_particles = np.sum(flat_hist)
    return total_particles / non_empty_cells

# -------------------------
# Function to list available simulations
# -------------------------
def list_available_simulations(outputs_folder):
    """List all available simulation folders and their parameters."""
    available_sims = []
    for folder in os.listdir(outputs_folder):
        folder_path = os.path.join(outputs_folder, folder)
        if not os.path.isdir(folder_path):
            continue
            
        sim_params = parse_simulation_folder(folder)
        if sim_params is None:
            continue
            
        particle_dist = get_particle_distribution(folder)
        available_sims.append({
            "folder": folder,
            "params": sim_params,
            "distribution": particle_dist
        })
    
    return available_sims

# ------------------------------------------------------
# Global parameters and directories
# ------------------------------------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))
temp_parent_dir = os.path.dirname(script_dir)
temp_temp_parent_dir = os.path.dirname(temp_parent_dir)
parent_dir = os.path.dirname(temp_temp_parent_dir)

# Search in Outputs/Binary C
outputs_folder = os.path.join(parent_dir, "Outputs", "Binary C")

pickle_folder = os.path.join(temp_temp_parent_dir, "Pickle Files")
save_location = os.path.join(temp_temp_parent_dir, "Error (CSV)", "Sensitivity Ranking (CSV)")
os.makedirs(save_location, exist_ok=True)


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

N_x = 14  
N_y = 6  
N_z = 4  
timestep_interval = 5e-6

# Update the x-direction grid so that it starts at x_offset instead of 0.
x_min = x_offset
x_max = barrel_length
x_edges = np.linspace(x_min, x_max, N_x + 1)
# For y_edges, use a fixed width based on the barrel geometry (using "1PD" as reference)
y_edges = np.linspace(-((axis_separation_mm["1PD"]/1000.0 + 2*outer_radius)/2.0),
                      ((axis_separation_mm["1PD"]/1000.0 + 2*outer_radius)/2.0), N_y + 1)
z_edges = np.linspace(-outer_radius, outer_radius, N_z + 1)

outlet_x_index = N_x - 1

all_simulation_data = []
all_mixing_metric_series = {}
simulation_results = {}  # Store results by folder

# -------------------------
# Main processing loop
# -------------------------
# List available simulations if needed
if not analyze_all_simulations and not selected_simulations:
    print("\nNo simulations selected. Listing available simulations:")
    available_sims = list_available_simulations(outputs_folder)
    
    if not available_sims:
        print("No valid simulation folders found.")
        exit(1)
    
    print("\nAvailable simulations:")
    for i, sim in enumerate(available_sims):
        params = sim["params"]
        print(f"{i+1}. {sim['folder']} ({sim['distribution']})")
        print(f"   - Screw spacing: {params['screw_spacing']}")
        print(f"   - Rotations: {params['rotations']}")
        print(f"   - RPM: {params['rpm']}")
        print(f"   - Type: {params['simtype']}")
    
    print("\nPlease update the 'selected_simulations' list in the script with the folders you want to analyze.")
    print("Example: selected_simulations = ['sim_r0_r3_1PD_0.5N_200', 'sim_r0_r3_1PD_1.5N_1000']")
    exit(0)

# Print the current settings
print("\nCurrent settings:")
print(f"Analyze all simulations: {analyze_all_simulations}")
if not analyze_all_simulations:
    print(f"Selected simulations: {selected_simulations}")
print(f"Use saved data: {use_saved_data}")
print(f"Calculate sensitivity: {calculate_sensitivity}")
print(f"Sensitivity interval: {sensitivity_interval}")
print(f"Scale factors: {selected_scales}")
print(f"X-offset: {x_offset}")

for folder in os.listdir(outputs_folder):
    folder_path = os.path.join(outputs_folder, folder)
    if not os.path.isdir(folder_path):
        continue
        
    # Skip if not in the selected simulations list (when analyze_all_simulations is False)
    if not analyze_all_simulations and folder not in selected_simulations:
        print(f"Skipping folder '{folder}': not in selected simulations list.", flush=True)
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
    if use_saved_data and os.path.exists(sim_data_file):
        with open(sim_data_file, "rb") as f:
            simulation_time_series = pd.read_pickle(f)
        print(f"Loaded saved simulation data for {folder}")
    else:
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
        simulation_time_series.sort(key=lambda entry: entry["time"])

    simulation_results[folder] = {"params": sim_params, "time_series": simulation_time_series}
    
    # Save global data for CSV (one row per timestep).
    all_simulation_data = [row for row in all_simulation_data if row[0] != folder]
    for entry in simulation_time_series:
        row = [folder, entry["time"], entry["particle_count"], entry["mixing_metric"]]
        for scale in selected_scales:
            label = f"M_scale_{round(scale,2)}"
            row.append(None)
        for scale in selected_scales:
            label = f"Avg_Particles_scale_{round(scale,2)}"
            row.append(None)
        all_simulation_data.append(row)
    
    times = [entry["time"] for entry in simulation_time_series]
    mix_series = [entry["mixing_metric"] for entry in simulation_time_series]
    all_mixing_metric_series[folder] = (times, mix_series)
    
    # Sensitivity Analysis (only if enabled)
    if calculate_sensitivity:
        sensitivity_metric_ts_this_sim = {}
        sensitivity_avg_particles_ts_this_sim = {}
        for scale in selected_scales:
            ts = []
            avg_particles_ts = []
            new_Nx = max(2, int(round(scale * N_x)))
            new_Ny = max(2, int(round(scale * N_y)))
            new_Nz = max(2, int(round(scale * N_z)))
            new_x_edges = np.linspace(x_min, x_max, new_Nx + 1)
            new_y_edges = np.linspace(y_min, y_max, new_Ny + 1)
            new_z_edges = np.linspace(z_min, z_max, new_Nz + 1)
            last_value = None
            last_avg_particles = None
            for i, entry in enumerate(simulation_time_series):
                if i % sensitivity_interval == 0:
                    print(f"Processing step: {int(i/timestep_interval)} for scale: {scale}", flush=True)
                    df = entry["df"]
                    coords = np.vstack([df["x"], df["y"], df["z"]]).T
                    H_new, _ = np.histogramdd(coords, bins=[new_x_edges, new_y_edges, new_z_edges])
                    metric = compute_mixing_metric(H_new, df, new_x_edges, new_y_edges, new_z_edges)
                    avg_particles = compute_avg_particles_per_cell(H_new)
                    last_value = metric
                    last_avg_particles = avg_particles
                    ts.append(metric)
                    avg_particles_ts.append(avg_particles)
                else:
                    ts.append(last_value)
                    avg_particles_ts.append(last_avg_particles)
            sensitivity_metric_ts_this_sim[scale] = ts
            sensitivity_avg_particles_ts_this_sim[scale] = avg_particles_ts
        
        # Update the entries with the sensitivity metrics
        for i, entry in enumerate(simulation_time_series):
            for scale in selected_scales:
                label = f"M_scale_{round(scale,2)}"
                entry[label] = sensitivity_metric_ts_this_sim[scale][i]
                avg_particles_label = f"Avg_Particles_scale_{round(scale,2)}"
                entry[avg_particles_label] = sensitivity_avg_particles_ts_this_sim[scale][i]
        
        # Update the CSV data with the sensitivity metrics
        all_simulation_data = [row for row in all_simulation_data if row[0] != folder]
        for entry in simulation_time_series:
            row = [folder, entry["time"], entry["particle_count"], entry["mixing_metric"]]
            for scale in selected_scales:
                label = f"M_scale_{round(scale,2)}"
                row.append(entry[label])
            for scale in selected_scales:
                avg_particles_label = f"Avg_Particles_scale_{round(scale,2)}"
                row.append(entry[avg_particles_label])
            all_simulation_data.append(row)

# -------------------------
# Save combined simulation summary data into one CSV file
# -------------------------
base_columns = ["Simulation", "Time (s)", "Particle Count", "Mixing Metric M"]
all_columns = base_columns + sensitivity_scale_labels + sensitivity_avg_particles_labels
df_all = pd.DataFrame(all_simulation_data, columns=all_columns)
combined_csv_filename = os.path.join(save_location, "sensitivity_ranking_BinaryC.csv")
df_all.to_csv(combined_csv_filename, index=False)
print(f"Saved combined simulation summary CSV", flush=True)