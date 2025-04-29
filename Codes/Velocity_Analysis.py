import os
import sys
import re
import glob
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from scipy.interpolate import UnivariateSpline
import warnings
import pickle  # Added import for pickle
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# =====================================================
# Directories
# =====================================================
script_dir = os.path.dirname(os.path.realpath(__file__)) # Analysis/Codes/Velocity Analysis
parent_dir = os.path.dirname(script_dir) # Analysis/Codes
grandparent_dir = os.path.dirname(parent_dir) # Analysis
grandgrandparent_dir = os.path.dirname(grandparent_dir) # Root
outputs_folder = os.path.join(grandgrandparent_dir, "Outputs", "Run1")
velocity_results_dir = os.path.join(grandparent_dir, "Velocity Analysis")
os.makedirs(velocity_results_dir, exist_ok=True)

# Path to the pickle files directory
pickle_files_dir = os.path.join(grandparent_dir, "Pickle Files")
os.makedirs(pickle_files_dir, exist_ok=True)

# Set to True to use existing pickle files, False to process all simulations from scratch
use_saved_data = False  # Change this to False to process all simulations from scratch

# Path to the CSV file containing steady state times
segregation_analysis_dir = os.path.join(grandparent_dir, "Segregation Analysis")
steady_state_csv = os.path.join(segregation_analysis_dir, "all_simulation_data_plots_mcc.csv")

# Load steady state times from CSV
steady_state_times = {}
if os.path.exists(steady_state_csv):
    steady_state_df = pd.read_csv(steady_state_csv)
    for _, row in steady_state_df.iterrows():
        folder = row['Simulation']
        steady_time = row['Steady State Time (s)']
        steady_state_times[folder] = steady_time
else:
    print(f"Warning: Could not find steady state times file at {steady_state_csv}")
    sys.exit(1)

# =====================================================
# Options and Grid Definitions
# =====================================================

# Common option
x_offset = 0.05  # (m)

# Geometry parameters (common to both analyses)
barrel_length = 0.18       # (m)
screw_radius = 0.00916     # (m)
barrel_clearance = 0.00025 # (m)
outer_radius = screw_radius + barrel_clearance

# Dictionary for axis separation (in mm) based on PD value.
axis_separation_mm = {
    "1PD": 15.00347,
    "2PD": 15.56194,
    "3PD": 16.12041,
    "4PD": 16.67888,
    "5PD": 17.23735
}

# -----------------------
# Analysis 1 (Scatter Analysis) grid (for x and z only; y is computed per simulation)
# -----------------------
N_x_a1 = 233   # number of x bins
N_y_a1 = 64    # number of y bins (for steady-state check)
N_z_a1 = 34    # number of z bins

x_edges_a1 = np.linspace(x_offset, barrel_length, N_x_a1 + 1)
z_edges_a1 = np.linspace(-outer_radius, outer_radius, N_z_a1 + 1)

# -----------------------
# Analysis 2 (Heatmap Analysis) grid (for x and z only; y is computed per simulation)
# -----------------------
analysis2_N_x = 233  # bins in x-direction
analysis2_N_y = 64   # bins in y-direction (y_edges computed per simulation)
analysis2_N_z = 34   # bins in z-direction

x_edges_a2 = np.linspace(x_offset, barrel_length, analysis2_N_x + 1)
z_edges_a2 = np.linspace(-outer_radius, outer_radius, analysis2_N_z + 1)

# =====================================================
# Helper Functions
# =====================================================

def parse_simulation_folder(folder_name):
    """
    Parse folder name for poly-dispersed (MCC) simulations.
    Expected format: sim_MCC_{screw_spacing}_{rotations}_{rpm}
    e.g., "sim_MCC_2PD_1.0N_200"
    """
    mcc_pattern = r"sim_MCC_(?P<screw_spacing>\d+PD)_(?P<rotations>(?:[0-9.]+)?N)_(?P<rpm>\d+)"
    m = re.match(mcc_pattern, folder_name)
    if m:
        params = m.groupdict()
        params["simtype"] = "MCC"
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
        if "v" in mesh.point_data:
            arr = mesh.point_data["v"]
            if arr.ndim == 2 and arr.shape[1] == 3:
                df["vx"] = arr[:, 0]
                df["vy"] = arr[:, 1]
                df["vz"] = arr[:, 2]
        else:
            for key, arr in mesh.point_data.items():
                if arr.ndim > 1:
                    df[key] = arr[:, 0]
                else:
                    df[key] = arr
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compute_average_velocity(sim_time_series, steady_time):
    vx_vals, vy_vals, vz_vals = [], [], []
    for entry in sim_time_series:
        if entry["time"] >= steady_time:
            df = entry["df"]
            if all(col in df.columns for col in ["vx", "vy", "vz"]):
                vx_vals.append(df["vx"].mean())
                vy_vals.append(df["vy"].mean())
                vz_vals.append(df["vz"].mean())
    if len(vx_vals) == 0:
        return np.nan, np.nan, np.nan
    return np.mean(vx_vals), np.mean(vy_vals), np.mean(vz_vals)

def compute_depth_time_averaged_heatmap(simulation_time_series, steady_time, x_edges, y_edges, z_edges):
    """
    For each timestep after steady state, compute the 3D average velocity magnitude
    in each cell (using weighted 3D histograms) and then depth-average (over z).
    Time-average the resulting 2D maps to produce a final heatmap.
    
    Returns:
      heatmap_avg: 2D numpy array of shape (len(x_edges)-1, len(y_edges)-1)
    """
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    n_z = len(z_edges) - 1
    heatmaps = []
    for entry in simulation_time_series:
        if entry["time"] >= steady_time:
            df = entry["df"]
            if all(col in df.columns for col in ["vx", "vy", "vz"]):
                v_mag = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
                df = df.copy()
                df["v_mag"] = v_mag
                sum_v, _ = np.histogramdd(df[["x", "y", "z"]].values,
                                          bins=[x_edges, y_edges, z_edges],
                                          weights=df["v_mag"])
                count_v, _ = np.histogramdd(df[["x", "y", "z"]].values,
                                            bins=[x_edges, y_edges, z_edges])
                avg_v = np.divide(sum_v, count_v, out=np.zeros_like(sum_v), where=count_v>0)
                depth_avg = np.zeros((n_x, n_y))
                for i in range(n_x):
                    for j in range(n_y):
                        cell_vals = avg_v[i, j, :]
                        valid = count_v[i, j, :] > 0
                        if np.any(valid):
                            depth_avg[i, j] = np.mean(cell_vals[valid])
                        else:
                            depth_avg[i, j] = 0
                heatmaps.append(depth_avg)
    if len(heatmaps) == 0:
        return np.zeros((n_x, n_y))
    return np.mean(heatmaps, axis=0)

# =====================================================
# Main Processing Loop: Process each simulation.
# =====================================================

simulation_velocity_data = []  # for scatter analysis
simulation_heatmap_data = []   # for heatmap analysis

for folder in os.listdir(outputs_folder):
    folder_path = os.path.join(outputs_folder, folder)
    if not os.path.isdir(folder_path):
        continue
    sim_params = parse_simulation_folder(folder)
    if sim_params is None:
        print(f"Skipping folder '{folder}': not recognized as poly-dispersed.")
        continue

    # Parse operating conditions.
    screw_spacing_str = sim_params["screw_spacing"]  # e.g., "2PD"
    try:
        spacing_val = int(re.search(r"(\d+)", screw_spacing_str).group(1))
    except Exception as e:
        print(f"Error parsing screw spacing for {folder}: {e}")
        continue
    pitch_str = sim_params["rotations"].strip()
    match = re.search(r"([\d.]+)", pitch_str)
    pitch_val = float(match.group(1)) if match else 1.0
    rpm_val = int(sim_params["rpm"])
    
    # Check if pickle file exists for this simulation
    pickle_filename = f"{folder}_velocity_data.pkl"
    pickle_path = os.path.join(pickle_files_dir, pickle_filename)
    
    if use_saved_data and os.path.exists(pickle_path):
        print(f"Loading existing pickle file for {folder} from {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # Extract data from pickle file
            avg_vx = pickle_data["avg_vx"]
            avg_vy = pickle_data["avg_vy"]
            avg_vz = pickle_data["avg_vz"]
            heatmap = pickle_data["heatmap"]
            steady_time = pickle_data["steady_time"]
            
            # Add data to the lists for analysis
            simulation_velocity_data.append({
                "folder": folder,
                "screw_spacing": spacing_val,
                "pitch": pitch_val,
                "rpm": rpm_val,
                "mean_vx": avg_vx,
                "mean_vy": avg_vy,
                "mean_vz": avg_vz
            })
            
            simulation_heatmap_data.append({
                "folder": folder,
                "screw_spacing": spacing_val,
                "pitch": pitch_val,
                "rpm": rpm_val,
                "heatmap": heatmap
            })
            
            print(f"Successfully loaded data for {folder} from pickle file")
            continue  # Skip to the next simulation
        except Exception as e:
            print(f"Error loading pickle file for {folder}: {e}")
            print("Will process the simulation from scratch")
    
    # If we get here, either the pickle file doesn't exist or there was an error loading it
    # Compute simulation-specific y_edges based on PD value.
    axis_sep_mm_val = axis_separation_mm.get(screw_spacing_str, 15.00347)
    axis_sep_ref_sim = axis_sep_mm_val / 1000.0
    y_half_sim = (axis_sep_ref_sim + 2 * outer_radius) / 2.0
    y_edges_a1_sim = np.linspace(-y_half_sim, y_half_sim, N_y_a1 + 1)
    y_edges_a2_sim = np.linspace(-y_half_sim, y_half_sim, analysis2_N_y + 1)
    
    particle_files = get_all_particle_files(folder_path)
    if not particle_files:
        print(f"No particle files in folder '{folder}'. Skipping simulation.")
        continue

    print(f"Processing simulation: {folder} with {len(particle_files)} timesteps")
    simulation_time_series = []
    for vtk_file in particle_files:
        base = os.path.basename(vtk_file)
        m = re.search(r"particles_(\d+)\.vtk", base)
        step = int(m.group(1)) if m else 0
        time = step * 5e-6  # (s)
        print(f"Processing step {step} from {vtk_file}", flush=True)
        df = load_vtk_data(vtk_file)
        if df is None:
            print(f"Skipping {vtk_file} due to loading error.")
            continue
        df = df[df["x"] >= x_offset]
        coords = np.vstack([df["x"], df["y"], df["z"]]).T
        H, _ = np.histogramdd(coords, bins=[x_edges_a1, y_edges_a1_sim, z_edges_a1])
        simulation_time_series.append({
            "time": time,
            "particle_count": np.sum(H),
            "df": df
        })
    simulation_time_series.sort(key=lambda entry: entry["time"])
    
    # Get steady state time from the CSV file
    if folder not in steady_state_times:
        print(f"Warning: No steady state time found for {folder} in CSV file. Skipping simulation.")
        continue
    steady_time = steady_state_times[folder]
    
    avg_vx, avg_vy, avg_vz = compute_average_velocity(simulation_time_series, steady_time)
    simulation_velocity_data.append({
        "folder": folder,
        "screw_spacing": spacing_val,
        "pitch": pitch_val,
        "rpm": rpm_val,
        "mean_vx": avg_vx,
        "mean_vy": avg_vy,
        "mean_vz": avg_vz
    })
    
    heatmap = compute_depth_time_averaged_heatmap(simulation_time_series, steady_time,
                                                  x_edges_a2, y_edges_a2_sim, z_edges_a2)
    simulation_heatmap_data.append({
        "folder": folder,
        "screw_spacing": spacing_val,
        "pitch": pitch_val,
        "rpm": rpm_val,
        "heatmap": heatmap
    })
    
    # Create pickle file for this simulation
    pickle_data = {
        "folder": folder,
        "screw_spacing": spacing_val,
        "pitch": pitch_val,
        "rpm": rpm_val,
        "steady_time": steady_time,
        "avg_vx": avg_vx,
        "avg_vy": avg_vy,
        "avg_vz": avg_vz,
        "heatmap": heatmap
    }
    
    # Create pickle filename with "velocity" in it
    pickle_filename = f"{folder}_velocity_data.pkl"
    pickle_path = os.path.join(pickle_files_dir, pickle_filename)
    
    # Save the pickle file
    with open(pickle_path, 'wb') as f:
        pickle.dump(pickle_data, f)
    
    print(f"Saved velocity data pickle file for {folder} to {pickle_path}")

# =====================================================
# (A) Scatter Plot Analysis (Existing)
# =====================================================
results_by_rpm = {}
for rec in simulation_velocity_data:
    rpm = rec["rpm"]
    results_by_rpm.setdefault(rpm, []).append(rec)
unique_rpms = sorted(results_by_rpm.keys())

velocity_components = [
    ("mean_vx", "Mean x-velocity (m/s)"),
    ("mean_vy", "Mean y-velocity (m/s)"),
    ("mean_vz", "Mean z-velocity (m/s)")
]
num_rows = len(unique_rpms)
num_cols = len(velocity_components)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), constrained_layout=True)
axs = np.atleast_2d(axs)

# Define pitch to flights mapping
pitch_to_flights = {0.5: 10, 1.0: 20, 1.5: 30}

unique_pitches = sorted({rec["pitch"] for rec in simulation_velocity_data})

# Use the same color scheme as the axial mixing profile faceted plots
# Using a colorblind-friendly palette that's common in research papers
spacing_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
pitch_colors = {pitch: spacing_colors[i % len(spacing_colors)] for i, pitch in enumerate(unique_pitches)}

# Create a single legend for all subplots
legend_handles = []
legend_labels = []

for i, rpm in enumerate(unique_rpms):
    rpm_records = results_by_rpm[rpm]
    for j, (vel_key, vel_label) in enumerate(velocity_components):
        ax = axs[i, j]
        
        # Add grid to the plot
        ax.grid(True, linestyle='--', alpha=0.7)
        
        for rec in rpm_records:
            pitch = rec["pitch"]
            flights = pitch_to_flights.get(pitch, pitch * 20)  # Default to pitch * 20 if not in mapping
            scatter = ax.scatter(rec["screw_spacing"], rec[vel_key],
                               color=pitch_colors[pitch], s=50,
                               label=f"Flights: {flights}")
            
            # Add to legend only once
            if i == 0 and j == 0:
                if f"Flights: {flights}" not in legend_labels:
                    legend_handles.append(scatter)
                    legend_labels.append(f"Flights: {flights}")
        
        # Only show x-axis label on bottom row
        if i == num_rows - 1:
            ax.set_xlabel("Screw Spacing (PR)")
        else:
            ax.set_xlabel("")
            
        ax.set_ylabel(vel_label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Remove individual legends
        ax.legend().remove()
        
    axs[i, 0].text(-0.25, 0.5, f"RPM: {rpm}", transform=axs[i, 0].transAxes,
                   va="center", ha="right", fontsize=12, fontweight="bold", rotation=90)

# Sort legend by flights value
legend_data = list(zip(legend_handles, legend_labels))
legend_data.sort(key=lambda x: int(x[1].split(": ")[1]))
sorted_handles, sorted_labels = zip(*legend_data)

# Add a single legend at the bottom of the figure with reduced whitespace
fig.legend(sorted_handles, sorted_labels, loc='center', bbox_to_anchor=(0.5, -0.02), 
          ncol=len(sorted_handles), fontsize=10)

# Adjust layout to make room for the legend with reduced whitespace
plt.subplots_adjust(bottom=0.12)

output_filename_scatter = os.path.join(velocity_results_dir, "Global_Velocity_vs_Operating_Conditions.png")
plt.savefig(output_filename_scatter, bbox_inches='tight')
print(f"Saved scatter plot analysis to {output_filename_scatter}")
plt.close()

# =====================================================
# Create CSV file of velocity results
# =====================================================
# Create a DataFrame from the simulation_velocity_data
velocity_df = pd.DataFrame(simulation_velocity_data)

# Add a column for the number of flights based on pitch
velocity_df['flights'] = velocity_df['pitch'].map(pitch_to_flights)

# Create a dictionary to store heatmap data
heatmap_dict = {}
for rec in simulation_heatmap_data:
    folder = rec['folder']
    heatmap = rec['heatmap']
    
    # Flatten the heatmap and create column names for each cell
    n_rows, n_cols = heatmap.shape
    for i in range(n_rows):
        for j in range(n_cols):
            col_name = f'heatmap_cell_{i}_{j}'
            if folder not in heatmap_dict:
                heatmap_dict[folder] = {}
            heatmap_dict[folder][col_name] = heatmap[i, j]

# Convert heatmap dictionary to DataFrame
heatmap_df = pd.DataFrame.from_dict(heatmap_dict, orient='index')

# Merge velocity and heatmap DataFrames
combined_df = pd.merge(velocity_df, heatmap_df, left_on='folder', right_index=True)

# Reorder columns for better readability
# First, get the velocity columns
velocity_cols = ['folder', 'screw_spacing', 'pitch', 'flights', 'rpm', 'mean_vx', 'mean_vy', 'mean_vz']
# Then, get all heatmap columns
heatmap_cols = [col for col in combined_df.columns if col not in velocity_cols]
# Combine them in the desired order
combined_df = combined_df[velocity_cols + heatmap_cols]

# Save to CSV
csv_output_path = os.path.join(velocity_results_dir, "velocity_and_heatmap_results.csv")
combined_df.to_csv(csv_output_path, index=False)
print(f"Saved velocity and heatmap results to CSV file: {csv_output_path}")

# =====================================================
# (B) Composite Heatmap Analysis
# Group heatmaps by RPM, then by (screw_spacing, pitch).
# Create a composite figure (5 rows x 3 columns) for each RPM.
# =====================================================

grouped_by_rpm = defaultdict(dict)
for rec in simulation_heatmap_data:
    key = (rec["screw_spacing"], rec["pitch"])
    grouped_by_rpm[rec["rpm"]][key] = rec

unique_screw_spacings = sorted({rec["screw_spacing"] for rec in simulation_heatmap_data})
unique_pitches = sorted({rec["pitch"] for rec in simulation_heatmap_data})

# Define pitch to flights mapping
pitch_to_flights = {0.5: 10, 1.0: 20, 1.5: 30}

for rpm_val, rec_dict in grouped_by_rpm.items():
    n_rows = len(unique_screw_spacings)    # expected to be 5
    n_cols = len(unique_pitches)           # expected to be 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    
    # Determine local vmin/vmax for this composite.
    local_vmin = np.inf
    local_vmax = -np.inf
    for key, rec in rec_dict.items():
        hm = rec["heatmap"]
        local_vmin = min(local_vmin, np.nanmin(hm))
        local_vmax = max(local_vmax, np.nanmax(hm))
    
    for i, spacing in enumerate(unique_screw_spacings):
        for j, pitch in enumerate(unique_pitches):
            ax = axes[i, j]
            key = (spacing, pitch)
            if key in rec_dict:
                hm = rec_dict[key]["heatmap"]
                # Recompute y_edges for analysis2 based on screw spacing.
                axis_sep_mm_val = axis_separation_mm.get(f"{spacing}PD", 15.00347)
                axis_sep_ref_sim = axis_sep_mm_val / 1000.0
                y_half_sim = (axis_sep_ref_sim + 2 * outer_radius) / 2.0
                y_edges_a2_sim = np.linspace(-y_half_sim, y_half_sim, analysis2_N_y + 1)
                im = ax.imshow(hm.T, origin="lower",
                               extent=[x_edges_a2[0], x_edges_a2[-1], y_edges_a2_sim[0], y_edges_a2_sim[-1]],
                               aspect="auto", cmap="rainbow", vmin=local_vmin, vmax=local_vmax)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            # Add column header (pitch) in top row.
            if i == 0:
                flights = pitch_to_flights.get(pitch, pitch * 20)  # Default to pitch * 20 if not in mapping
                ax.set_title(f"{flights} Flights", fontsize=12, fontweight="bold")
            # Add row header (screw spacing) in first column.
            if j == 0:
                ax.set_ylabel(f"{spacing}PR", fontsize=12, fontweight="bold", rotation=0, labelpad=30, va="center")
    
    # Add an overall title for the composite.
    # fig.suptitle(f"Composite Heatmaps for RPM: {rpm_val}", fontsize=14)
    # Add one overarching colorbar to the right.
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical")
    cbar.set_label("Average particle velocity (m/s)")
    output_filename_composite = os.path.join(velocity_results_dir, f"Composite_Heatmaps_RPM_{rpm_val}.png")
    plt.savefig(output_filename_composite)
    print(f"Saved composite heatmap figure for RPM {rpm_val} to {output_filename_composite}")
    plt.close()
