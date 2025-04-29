import os
import re
import glob
import pickle
import sys
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import warnings
import gc  # Added for garbage collection
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec
from collections import defaultdict
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# ------------------------------------------------------
# Global Parameters and Directories
# ------------------------------------------------------
use_saved_data = True  # Set to True to load previously saved particle data
x_offset = 0.05         # x-offset for particle selection (in meters)
barrel_length = 0.18    # Barrel length (in meters)
screw_radius = 0.00916  # Screw radius (in meters)
barrel_clearance = 0.00025
outer_radius = screw_radius + barrel_clearance

# Grid parameters for spatial binning
N_x = 233  
N_y = 64  
N_z = 34  
timestep_interval = 5e-6  # timestep interval in seconds

x_min = x_offset
x_max = barrel_length
x_edges = np.linspace(x_min, x_max, N_x + 1)
axis_separation_mm = {"1PD": 15.00347}  # reference value for y_edges
y_half = (axis_separation_mm["1PD"] / 1000.0 + 2 * outer_radius) / 2.0
y_edges = np.linspace(-y_half, y_half, N_y + 1)
z_edges = np.linspace(-outer_radius, outer_radius, N_z + 1)

# Directories for input and output
script_dir = os.path.dirname(os.path.realpath(__file__)) # Analysis/Codes/Force Analysis
parent_dir = os.path.dirname(script_dir) # Analysis/Codes
grandparent_dir = os.path.dirname(parent_dir) # Analysis
project_root = os.path.dirname(grandparent_dir) # Research Project Code root
outputs_folder = os.path.join(project_root, "Outputs", "Run1") # Root/Outputs/Run1
save_csv_location = os.path.join(grandparent_dir, "Force Analysis")
force_plots_dir = os.path.join(grandparent_dir, "Force Analysis")
pickle_folder = os.path.join(grandparent_dir, "Pickle Files") # Store pickle files in Analysis/Pickle Files
os.makedirs(save_csv_location, exist_ok=True)
os.makedirs(force_plots_dir, exist_ok=True)
os.makedirs(pickle_folder, exist_ok=True)

# Path to the mixing metric CSV (for steady state determination)
mixing_csv_path = os.path.join(grandparent_dir, "Segregation Analysis", "all_simulation_data_plots_mcc.csv")

# Steady state configuration
window_steady = 5  # number of timesteps in sliding window
tol = 0.0075        # tolerance for steady state (relative change)
ignore_time = 2.0   # ignore mixing metric data before this time (in seconds)

# Define pitch to flights mapping
pitch_to_flights = {0.5: 10, 1.0: 20, 1.5: 30}

# ------------------------------------------------------
# Function: Parse Simulation Folder Name
# ------------------------------------------------------
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

# ------------------------------------------------------
# Function: Get All Particle Files (particles_*.vtk)
# ------------------------------------------------------
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

# ------------------------------------------------------
# Function: Load VTK Data for Particles
# ------------------------------------------------------
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
            elif key.lower() == "f" and arr.ndim == 2 and arr.shape[1] == 3:
                df["fx"] = arr[:, 0]
                df["fy"] = arr[:, 1]
                df["fz"] = arr[:, 2]
            else:
                if arr.ndim > 1:
                    df[key] = arr[:, 0]
                else:
                    df[key] = arr
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}", flush=True)
        return None

# ------------------------------------------------------
# Function: Bin Data Along x-direction
# ------------------------------------------------------
def bin_data_along_x(df, x_edges, value_column):
    bin_means = np.zeros(len(x_edges) - 1)
    for i in range(len(x_edges) - 1):
        mask = (df["x"] >= x_edges[i]) & (df["x"] < x_edges[i + 1])
        if mask.sum() > 0:
            bin_means[i] = df.loc[mask, value_column].mean()
        else:
            bin_means[i] = np.nan
    return bin_means

# ------------------------------------------------------
# Analysis 1: Axial Force Profile (Particle-based)
# ------------------------------------------------------
def analyze_axial_force_profile(simulation_time_series, x_edges):
    """
    Computes the axial profile of the average force magnitude on particles.
    For each timestep (after steady state), computes:
        |F| = sqrt(fx^2 + fy^2 + fz^2)
    Then, bins these values along the x-direction using x_edges.
    The profiles are averaged over all timesteps.
    Returns log10 of the force values.
    """
    axial_profiles = []
    total_steps = len(simulation_time_series)
    for idx, entry in enumerate(simulation_time_series):
        df = entry["df"]
        if not all(col in df.columns for col in ["fx", "fy", "fz"]):
            continue
        df["force_mag"] = np.sqrt(df["fx"]**2 + df["fy"]**2 + df["fz"]**2)
        profile = bin_data_along_x(df, x_edges, "force_mag")
        # Convert to log10 scale, handling zeros and negative values
        profile = np.log10(np.maximum(profile, 1e-10))  # Use small positive number to avoid log(0)
        axial_profiles.append(profile)
    if len(axial_profiles) == 0:
        return None
    axial_profiles = np.stack(axial_profiles, axis=0)
    avg_profile = np.nanmean(axial_profiles, axis=0)
    return avg_profile

# ------------------------------------------------------
# Analysis 2: 2D Heatmap of Contact Forces (Depth- and Time-Averaged)
# ------------------------------------------------------
def get_time_from_filename(filename, prefix="pairs_", suffix=".vtk"):
    """
    Extracts the timestep number from the filename (assumes pattern like pairs_123456.vtk)
    and returns the corresponding simulation time.
    """
    base = os.path.basename(filename)
    m = re.search(r"pairs_(\d+)\.vtk", base)
    if m:
        step = int(m.group(1))
        return step * timestep_interval
    return None

def analyze_contact_force_heatmap(sim_folder, steady_state_time, x_edges, y_edges):
    """
    Processes contact force VTK files (pairs_*.vtk) for timesteps >= steady_state_time.
    For each such file, computes the cell centers (representing contact locations),
    extracts the total contact force from the cell data key "force" (or combines force_normal
    and force_tangential if "force" is unavailable), and calculates the force magnitude.
    The contacts are then binned into an x-y grid (averaging over the z-axis).
    Finally, the heatmaps from all qualifying timesteps are averaged to produce a final 2D heatmap.
    """
    contact_pattern = os.path.join(sim_folder, "pairs_*.vtk")
    contact_files = glob.glob(contact_pattern)
    if not contact_files:
        print(f"No contact files found in {sim_folder}", flush=True)
        return None

    # Initialize heatmap sum and count arrays
    heatmap_sum = np.zeros((len(x_edges)-1, len(y_edges)-1))
    heatmap_count = np.zeros((len(x_edges)-1, len(y_edges)-1))
    
    total_steps = len(contact_files)
    for idx, file in enumerate(contact_files):
        print(f"Processing contact file for heatmap: Step {idx+1} of {total_steps}", flush=True)
        time_val = get_time_from_filename(file)
        if time_val is None or time_val < steady_state_time:
            continue  # Skip files before steady state

        try:
            mesh = pv.read(file)
            centers = mesh.cell_centers().points  # (N, 3) array for cell centers
            # Extract total contact force from cell_data
            if "force" in mesh.cell_data:
                force_vec = mesh.cell_data["force"]
                fmag = np.linalg.norm(force_vec, axis=1)
            elif "force_normal" in mesh.cell_data and "force_tangential" in mesh.cell_data:
                fn = mesh.cell_data["force_normal"]
                ft = mesh.cell_data["force_tangential"]
                fmag = np.sqrt(np.linalg.norm(fn, axis=1)**2 + np.linalg.norm(ft, axis=1)**2)
            else:
                print(f"Warning: No suitable force field found in {file}.", flush=True)
                continue

            # Bin the force magnitudes into x-y bins (averaging over z)
            x = centers[:, 0]
            y = centers[:, 1]
            
            # Process in batches to reduce memory usage
            batch_size = 10000
            for i in range(0, len(x), batch_size):
                end_idx = min(i + batch_size, len(x))
                x_batch = x[i:end_idx]
                y_batch = y[i:end_idx]
                f_batch = fmag[i:end_idx]
                
                for xi, yi, f in zip(x_batch, y_batch, f_batch):
                    i_x = np.searchsorted(x_edges, xi) - 1
                    j_y = np.searchsorted(y_edges, yi) - 1
                    if i_x >= 0 and i_x < heatmap_sum.shape[0] and j_y >= 0 and j_y < heatmap_sum.shape[1]:
                        heatmap_sum[i_x, j_y] += f
                        heatmap_count[i_x, j_y] += 1
            
            # Clear memory
            del mesh, centers, force_vec, fmag, x, y
            gc.collect()
            
        except Exception as e:
            print(f"Error processing contact file {file}: {e}", flush=True)
            continue

    # Calculate average heatmap
    with np.errstate(invalid='ignore'):
        avg_heatmap = np.divide(heatmap_sum, heatmap_count, out=np.zeros_like(heatmap_sum), where=heatmap_count!=0)
    
    return avg_heatmap

# ------------------------------------------------------
# Function: Get Steady State Time from Mixing Metric CSV
# ------------------------------------------------------
def get_steady_state_time(simulation_name):
    """
    Loads the mixing metric CSV and retrieves the steady state time for the given simulation.
    """
    try:
        df = pd.read_csv(mixing_csv_path)
    except Exception as e:
        print(f"Error loading mixing metric CSV: {e}", flush=True)
        return None

    df_sim = df[df["Simulation"] == simulation_name]
    if df_sim.empty:
        print(f"No mixing metric data found for simulation {simulation_name}", flush=True)
        return None

    # Get the steady state time from the CSV
    steady_time = df_sim["Steady State Time (s)"].iloc[0]
    if pd.isna(steady_time):
        print(f"No steady state time found for simulation {simulation_name}", flush=True)
        return None

    print(f"Using steady state time for {simulation_name}: {steady_time:.2f} s", flush=True)
    return steady_time

# ------------------------------------------------------
# Function: Process a single simulation
# ------------------------------------------------------
def process_simulation(folder, outputs_folder, pickle_folder, x_edges, y_edges):
    """
    Process a single simulation and return the results.
    This function is designed to be called for each simulation separately
    to reduce memory usage.
    """
    folder_path = os.path.join(outputs_folder, folder)
    if not os.path.isdir(folder_path):
        return None, None, None, None, None

    sim_params = parse_simulation_folder(folder)
    if sim_params is None:
        print(f"Skipping folder '{folder}': simulation parameters not recognized.", flush=True)
        return None, None, None, None, None

    # Determine steady state time using the mixing metric CSV
    steady_time = get_steady_state_time(folder)
    if steady_time is None:
        print(f"Steady state not reached for simulation {folder}. Using last timestep value instead.", flush=True)
        # We'll set steady_time later after loading the particle files
        steady_time = None
    
    # Process particle files for particle-based analyses
    particle_files = get_all_particle_files(folder_path)
    if not particle_files:
        print(f"No particle files found in folder '{folder}'. Skipping simulation.", flush=True)
        return None, None, None, None, None

    print(f"Processing simulation: {folder} with {len(particle_files)} particle timesteps.", flush=True)
    sim_data_file = os.path.join(pickle_folder, f"{folder}_force_data.pkl")
    simulation_time_series = []
    
    # Load or create simulation time series
    if use_saved_data and os.path.exists(sim_data_file):
        with open(sim_data_file, "rb") as f:
            simulation_time_series = pd.read_pickle(f)
        print(f"Loaded saved particle force data for {folder}", flush=True)
    else:
        for vtk_file in particle_files:
            m = re.search(r"particles_(\d+)\.vtk", os.path.basename(vtk_file))
            step = int(m.group(1)) if m else 0
            time = step * timestep_interval
            print(f"Processing particle file: {vtk_file} (Step {step})", flush=True)
            df = load_vtk_data(vtk_file)
            if df is None:
                continue
            df = df[df["x"] >= x_offset]
            if not all(col in df.columns for col in ["fx", "fy", "fz"]):
                print(f"Force data missing in file: {vtk_file}", flush=True)
                continue
            simulation_time_series.append({
                "time": time,
                "df": df,
                "particle_count": len(df),
                "mixing_metric": np.nan  # Placeholder; actual mixing metric comes from CSV
            })
        simulation_time_series.sort(key=lambda entry: entry["time"])
        with open(sim_data_file, "wb") as f:
            pickle.dump(simulation_time_series, f)
        print(f"Saved particle force data for {folder} to {sim_data_file}", flush=True)

    # If steady_time is None, use the last timestep value
    if steady_time is None:
        steady_time = simulation_time_series[-1]["time"]
        print(f"Using last timestep ({steady_time:.2f} s) as steady state time for {folder}", flush=True)

    # Filter particle data to only include timesteps at or after steady state
    steady_series = [entry for entry in simulation_time_series if entry["time"] >= steady_time]
    if not steady_series:
        print(f"No particle data available after steady state for {folder}. Skipping simulation.", flush=True)
        return None, None, None, None, None

    # ---- Analysis 1: Axial Force Profile (Particle-based) ----
    avg_profile = analyze_axial_force_profile(steady_series, x_edges)
    
    # ---- Analysis 2: 2D Heatmap of Contact Forces (Depth- and Time-Averaged) ----
    heatmap_data_file = os.path.join(pickle_folder, f"{folder}_heatmap_data.pkl")
    if use_saved_data and os.path.exists(heatmap_data_file):
        with open(heatmap_data_file, "rb") as f:
            heatmap = pickle.load(f)
        print(f"Loaded saved contact force heatmap data for {folder}", flush=True)
    else:
        heatmap = analyze_contact_force_heatmap(folder_path, steady_time, x_edges, y_edges)
        if heatmap is not None:
            with open(heatmap_data_file, "wb") as f:
                pickle.dump(heatmap, f)
            print(f"Saved contact force heatmap data for {folder} to {heatmap_data_file}", flush=True)
    
    # Extract simulation parameters
    spacing_val = int(re.search(r"(\d+)", sim_params["screw_spacing"]).group(1))
    screw_pitch = sim_params["rotations"].strip()
    match = re.search(r"([\d.]+)N", screw_pitch)
    pitch_val = float(match.group(1)) if match else 1.0
    rpm_val = int(sim_params["rpm"])
    sim_type = sim_params["simtype"]
    
    # Clear memory
    del simulation_time_series, steady_series
    gc.collect()
    
    return avg_profile, heatmap, sim_params, spacing_val, pitch_val, rpm_val, sim_type, steady_time

# ------------------------------------------------------
# Function: Create faceted axial force profile plots
# ------------------------------------------------------
def create_faceted_axial_force_plots(df_axial_forces, force_plots_dir):
    """
    Create faceted axial force profile plots for each simulation type.
    """
    if df_axial_forces.empty:
        return
    
    # Group by simulation type
    for sim_type in df_axial_forces["Simulation Type"].unique():
        df_type = df_axial_forces[df_axial_forces["Simulation Type"] == sim_type]
        
        # Get unique values
        unique_rpms = sorted(df_type["RPM"].unique())
        unique_pitches = sorted(df_type["Pitch (N)"].unique())
        unique_spacings = sorted(df_type["Screw Spacing (PD)"].unique())
        
        # Create figure with a grid for RPM (rows) and Pitch (columns)
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(len(unique_rpms), len(unique_pitches), figure=fig)
        
        # Define colors for spacing values - using a more research-paper-friendly color palette
        # Using a colorblind-friendly palette that's common in research papers
        spacing_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
        
        # Create legend handles and labels
        legend_handles = []
        legend_labels = []
        
        # Create subplots for each RPM-Pitch combination
        for i, rpm in enumerate(unique_rpms):
            for j, pitch in enumerate(unique_pitches):
                ax = fig.add_subplot(gs[i, j])
                
                # Filter data for this subplot
                subplot_data = df_type[
                    (df_type["RPM"] == rpm) &
                    (df_type["Pitch (N)"] == pitch)
                ]
                
                # Plot each spacing value
                for spacing_idx, spacing in enumerate(unique_spacings):
                    data = subplot_data[subplot_data["Screw Spacing (PD)"] == spacing]
                    
                    if not data.empty:
                        line = ax.plot(data["Axial Position (m)"], data["Avg Particle Force (N)"],
                               color=spacing_colors[spacing_idx % len(spacing_colors)],
                               linewidth=2,
                               label=f'{spacing} PR')
                        
                        # Add to legend handles if we haven't seen this spacing before
                        if spacing not in [int(label.split()[0]) for label in legend_labels]:
                            legend_handles.append(line[0])
                            legend_labels.append(f'{spacing} PR')
                
                # Set labels and title
                ax.set_xlabel('Axial Position (m)' if i == len(unique_rpms)-1 else '')
                ax.set_ylabel('log₁₀(Average Particle Force) [N]' if j == 0 else '')
                
                ax.grid(True, alpha=0.3)
        
        # Add overarching headers for RPM values (rows)
        num_rows = len(unique_rpms)
        for i, rpm in enumerate(unique_rpms):
            # Calculate the vertical center position for the RPM header
            center_y = 1 - (i + 0.5) / num_rows  # Center the RPM header based on the number of rows
            fig.text(-0.02, center_y, f'{rpm} RPM', ha='center', va='center', fontsize=16, fontweight='bold', rotation=90)
        
        # Add overarching headers for Pitch values (columns)
        num_columns = len(unique_pitches)
        for j, pitch in enumerate(unique_pitches):
            # Calculate the horizontal center position for the Pitch header
            center_x = (j + 0.5) / num_columns  # Center the Pitch header based on the number of columns
            # Convert pitch to number of flights (1N = 20 flights)
            num_flights = int(pitch * 20)
            fig.text(center_x, 0.98, f'{num_flights} Flights', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Add legend at the bottom of the figure
        fig.legend(handles=legend_handles,
                  labels=legend_labels,
                  loc='center',
                  bbox_to_anchor=(0.5, 0.02),
                  title='Screw Spacing (PR = 0.55847mm)',
                  fontsize=12,
                  title_fontsize=14,
                  ncol=len(unique_spacings))  # Arrange items horizontally
        
        # Adjust layout to make room for legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure
        plt.savefig(os.path.join(force_plots_dir, f'axial_force_profile_faceted_{sim_type}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created faceted axial force profile plot for {sim_type}", flush=True)
        
        # Clear memory
        gc.collect()

# ------------------------------------------------------
# Function: Create composite axial force profile plot
# ------------------------------------------------------
def create_composite_axial_force_plot(df_axial_forces, force_plots_dir):
    """
    Create a composite axial force profile plot.
    """
    if df_axial_forces.empty:
        return
    
    # Add distribution type column
    df_axial_forces['Distribution_Type'] = df_axial_forces['Simulation Type'].apply(
        lambda x: 'Poly-dispersed' if x == 'MCC' else f'Binary {x}')
    
    # Get unique values for subplots
    distributions = ['Binary A', 'Binary B', 'Binary C', 'Poly-dispersed']
    spacings = sorted(df_axial_forces['Screw Spacing (PD)'].unique())
    
    # Create figure
    fig = plt.figure(figsize=(25, 16))
    gs = GridSpec(4, 5, figure=fig)
    
    # Define specific colors for RPM values - using a more research-paper-friendly color palette
    unique_rpms = sorted(df_axial_forces['RPM'].unique())
    unique_pitches = sorted(df_axial_forces['Pitch (N)'].unique())
    # Using a colorblind-friendly palette that's common in research papers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Define line styles for different pitches
    line_styles = {
        0.5: ('-', 2.25),      # Solid line, thicker
        1.0: ('--', 2.25),     # Dashed line, thicker
        1.5: (':', 2.25)       # Dotted line, thicker
    }
    
    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    added_combinations = set()
    
    # Create subplots
    for i, dist in enumerate(distributions):
        # Calculate the vertical center position for the header
        center_y = (len(distributions) - i - 0.5) / len(distributions)  # Center the header based on the number of distributions
        fig.text(-0.02, center_y, dist, ha='center', va='center', fontsize=16, fontweight='bold', rotation=90)
        
        for j, spacing in enumerate(spacings):
            ax = fig.add_subplot(gs[i, j])
            
            # Filter data for this subplot
            subplot_data = df_axial_forces[
                (df_axial_forces['Distribution_Type'] == dist) & 
                (df_axial_forces['Screw Spacing (PD)'] == spacing)
            ]
            
            # Plot each RPM-Pitch combination
            for rpm_idx, rpm in enumerate(unique_rpms):
                for pitch in unique_pitches:
                    data = subplot_data[
                        (subplot_data['RPM'] == rpm) & 
                        (subplot_data['Pitch (N)'] == pitch)
                    ]
                    
                    if not data.empty:
                        style, width = line_styles[pitch]
                        line = ax.plot(data['Axial Position (m)'], data['Avg Particle Force (N)'],
                                     color=colors[rpm_idx],
                                     linestyle=style,
                                     linewidth=width)
                        
                        # Add to legend if we haven't seen this combination before
                        combination = f'{rpm} RPM, {pitch}N'
                        if combination not in added_combinations:
                            legend_handles.append(line[0])
                            legend_labels.append(combination)
                            added_combinations.add(combination)
            
            # Set labels and title
            ax.set_xlabel('Axial Position (m)' if i == 3 else '')
            ax.set_ylabel('log₁₀(Average Particle Force) [N]' if j == 0 else '')
            ax.grid(True, alpha=0.3)
    
    # Add overarching headers for particle diameter values
    num_columns = len(spacings)  # Get the number of spacing columns
    for j, spacing in enumerate(spacings):
        # Calculate the horizontal center position for the spacing header
        center_x = (j + 0.5) / num_columns  # Center the spacing header based on the number of columns
        # Show the number of particle diameters (PD)
        fig.text(center_x, 1.015, f'{spacing} Particle Radii', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add single legend outside the subplots
    legend = fig.legend(handles=legend_handles,
                       labels=legend_labels,
                       loc='center', 
                       bbox_to_anchor=(0.5, -0.02),  # Moved even closer to plots
                       title='Operating Conditions',
                       fontsize=12,
                       title_fontsize=14,
                       ncol=3)  # Arrange in 3 columns
    
    # Ensure legend text is visible
    plt.setp(legend.get_texts(), color='black')
    plt.setp(legend.get_title(), color='black')
    
    # Adjust layout to prevent overlap and make room for legend
    plt.tight_layout(rect=[0, 0.02, 1, 1])  # Further reduced bottom margin
    
    # Save figure
    plt.savefig(os.path.join(force_plots_dir, 'axial_force_profile_composite.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Created composite axial force profile plot", flush=True)
    
    # Clear memory
    gc.collect()

# ------------------------------------------------------
# Function: Create faceted heatmap plots
# ------------------------------------------------------
def create_faceted_heatmap_plots(df_heatmap, force_plots_dir):
    """
    Create faceted heatmap plots for each simulation type and RPM value.
    """
    if df_heatmap.empty:
        return
    
    # Group by simulation type
    for sim_type in df_heatmap["Simulation Type"].unique():
        df_type = df_heatmap[df_heatmap["Simulation Type"] == sim_type]
        
        # Group by RPM
        grouped_by_rpm = defaultdict(dict)
        for _, row in df_type.iterrows():
            key = (row["Screw Spacing (PD)"], row["Pitch (N)"])
            grouped_by_rpm[row["RPM"]][key] = row
        
        unique_screw_spacings = sorted(df_type["Screw Spacing (PD)"].unique())
        unique_pitches = sorted(df_type["Pitch (N)"].unique())
        
        # Create a composite heatmap for each RPM value
        for rpm_val, rec_dict in grouped_by_rpm.items():
            n_rows = len(unique_screw_spacings)    # expected to be 5
            n_cols = len(unique_pitches)           # expected to be 3
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows),
                                     constrained_layout=True, sharex=True, sharey=True)
            axes = np.atleast_2d(axes)
            
            # Determine local vmin/vmax for this composite
            local_vmin = np.inf
            local_vmax = -np.inf
            for key, row in rec_dict.items():
                # Create a heatmap from the data
                x_unique = sorted(df_type[df_type["Simulation"] == row["Simulation"]]["x (m)"].unique())
                y_unique = sorted(df_type[df_type["Simulation"] == row["Simulation"]]["y (m)"].unique())
                X, Y = np.meshgrid(x_unique, y_unique)
                Z = np.zeros((len(x_unique), len(y_unique)))
                
                # Fill the Z array with force values
                for _, data_row in df_type[df_type["Simulation"] == row["Simulation"]].iterrows():
                    x_idx = x_unique.index(data_row["x (m)"])
                    y_idx = y_unique.index(data_row["y (m)"])
                    Z[x_idx, y_idx] = data_row["Avg Contact Force (N)"]
                
                # Convert to log10 scale, handling zeros and negative values
                Z_log = np.log10(np.maximum(Z, 1e-10))  # Use small positive number to avoid log(0)
                local_vmin = min(local_vmin, np.nanmin(Z_log))
                local_vmax = max(local_vmax, np.nanmax(Z_log))
            
            # Create the heatmaps
            for i, spacing in enumerate(unique_screw_spacings):
                for j, pitch in enumerate(unique_pitches):
                    ax = axes[i, j]
                    key = (spacing, pitch)
                    
                    if key in rec_dict:
                        row = rec_dict[key]
                        # Create a heatmap from the data
                        x_unique = sorted(df_type[df_type["Simulation"] == row["Simulation"]]["x (m)"].unique())
                        y_unique = sorted(df_type[df_type["Simulation"] == row["Simulation"]]["y (m)"].unique())
                        X, Y = np.meshgrid(x_unique, y_unique)
                        Z = np.zeros((len(x_unique), len(y_unique)))
                        
                        # Fill the Z array with force values
                        for _, data_row in df_type[df_type["Simulation"] == row["Simulation"]].iterrows():
                            x_idx = x_unique.index(data_row["x (m)"])
                            y_idx = y_unique.index(data_row["y (m)"])
                            Z[x_idx, y_idx] = data_row["Avg Contact Force (N)"]
                        
                        # Convert to log10 scale
                        Z_log = np.log10(np.maximum(Z, 1e-10))  # Use small positive number to avoid log(0)
                        
                        # Recompute y_edges for this specific screw spacing
                        axis_sep_mm_val = axis_separation_mm.get(f"{spacing}PD", 15.00347)
                        axis_sep_ref_sim = axis_sep_mm_val / 1000.0
                        y_half_sim = (axis_sep_ref_sim + 2 * outer_radius) / 2.0
                        y_edges_sim = np.linspace(-y_half_sim, y_half_sim, N_y + 1)
                        
                        im = ax.imshow(Z_log.T, origin="lower",
                                       extent=[x_edges[0], x_edges[-1], y_edges_sim[0], y_edges_sim[-1]],
                                       aspect="auto", cmap="rainbow", vmin=local_vmin, vmax=local_vmax)
                    else:
                        ax.axis("off")
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Add column header (pitch) in top row
                    if i == 0:
                        flights = pitch_to_flights.get(pitch, pitch * 20)  # Default to pitch * 20 if not in mapping
                        ax.set_title(f"{flights} Flights", fontsize=12, fontweight="bold")
                    
                    # Add row header (screw spacing) in first column
                    if j == 0:
                        ax.set_ylabel(f"{spacing}PR", fontsize=12, fontweight="bold", rotation=0, labelpad=30, va="center")
            
            # Add one overarching colorbar to the right
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical")
            cbar.set_label("log₁₀(Average Contact Force) [N]")
            
            # Save the figure
            plt.savefig(os.path.join(force_plots_dir, f"contact_force_heatmap_composite_{sim_type}_RPM_{rpm_val}.png"))
            plt.close()
            print(f"Created composite contact force heatmap plot for {sim_type} at RPM {rpm_val}", flush=True)
            
            # Clear memory
            gc.collect()

# ------------------------------------------------------
# Main Processing Loop for Force Analyses
# ------------------------------------------------------
def main():
    # Initialize data structures
    all_axial_rows = []       # For storing axial profile data for faceted plots
    all_heatmap_rows = []     # For storing heatmap data for faceted plots
    
    # Get list of simulation folders
    simulation_folders = [folder for folder in os.listdir(outputs_folder) 
                         if os.path.isdir(os.path.join(outputs_folder, folder))]
    
    # Process each simulation folder
    for folder in simulation_folders:
        print(f"\n{'='*80}\nProcessing simulation: {folder}\n{'='*80}", flush=True)
        
        # Process the simulation
        result = process_simulation(folder, outputs_folder, pickle_folder, x_edges, y_edges)
        if result is None:
            continue
            
        avg_profile, heatmap, sim_params, spacing_val, pitch_val, rpm_val, sim_type, steady_time = result
        
        # Process axial force profile data
        if avg_profile is not None:
            # Store data for faceted plots
            x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2.0
            for pos, force_val in zip(x_midpoints, avg_profile):
                all_axial_rows.append([
                    folder, sim_type, spacing_val, pitch_val, rpm_val, 
                    pos, force_val, True, steady_time
                ])
        else:
            print(f"No axial force profile generated for {folder}.", flush=True)

        # Process heatmap data
        if heatmap is not None:
            # Store data for faceted heatmaps
            x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2.0
            y_midpoints = (y_edges[:-1] + y_edges[1:]) / 2.0
            X, Y = np.meshgrid(x_midpoints, y_midpoints, indexing='ij')
            
            for i_x in range(X.shape[0]):
                for i_y in range(Y.shape[1]):
                    all_heatmap_rows.append([
                        folder, sim_type, spacing_val, pitch_val, rpm_val,
                        X[i_x, i_y], Y[i_x, i_y], heatmap[i_x, i_y], True, steady_time
                    ])
        else:
            print(f"No contact force heatmap generated for {folder}.", flush=True)
        
        # Clear memory after each simulation
        gc.collect()
    
    # Save summary CSVs
    if all_axial_rows:
        # Save axial force profile summary CSV
        df_axial_forces = pd.DataFrame(all_axial_rows, 
            columns=["Simulation", "Simulation Type", "Screw Spacing (PD)", "Pitch (N)", "RPM", 
                     "Axial Position (m)", "Avg Particle Force (N)", "Steady State Reached", "Steady State Time (s)"])
        df_axial_forces.to_csv(os.path.join(save_csv_location, "axial_force_profile_data.csv"), index=False)
        print("Saved axial force profile summary CSV.", flush=True)
        
        # Create faceted axial force profile plots
        create_faceted_axial_force_plots(df_axial_forces, force_plots_dir)
        
        # Create composite axial force profile plot
        create_composite_axial_force_plot(df_axial_forces, force_plots_dir)
    else:
        print("No axial force profile data to save.", flush=True)
    
    if all_heatmap_rows:
        # Save contact force heatmap data CSV
        df_heatmap = pd.DataFrame(all_heatmap_rows, 
            columns=["Simulation", "Simulation Type", "Screw Spacing (PD)", "Pitch (N)", "RPM", 
                     "x (m)", "y (m)", "Avg Contact Force (N)", "Steady State Reached", "Steady State Time (s)"])
        df_heatmap.to_csv(os.path.join(save_csv_location, "contact_force_heatmap_data.csv"), index=False)
        print("Saved contact force heatmap summary CSV.", flush=True)
        
        # Create faceted heatmap plots
        create_faceted_heatmap_plots(df_heatmap, force_plots_dir)
    else:
        print("No contact force heatmap data to save.", flush=True)
    
    print("Force analysis complete.", flush=True)

if __name__ == "__main__":
    main()
