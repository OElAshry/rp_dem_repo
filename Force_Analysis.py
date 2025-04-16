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
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# Global Parameters
use_saved_data = False
x_offset = 0.05
barrel_length = 0.18
screw_radius = 0.00916
barrel_clearance = 0.00025
outer_radius = screw_radius + barrel_clearance

# Grid parameters
N_x = 130  
N_y = 36  
N_z = 19  
timestep_interval = 5e-6

x_min = x_offset
x_max = barrel_length
x_edges = np.linspace(x_min, x_max, N_x + 1)
axis_separation_mm = {"1PD": 15.00347}
y_half = (axis_separation_mm["1PD"] / 1000.0 + 2 * outer_radius) / 2.0
y_edges = np.linspace(-y_half, y_half, N_y + 1)
z_edges = np.linspace(-outer_radius, outer_radius, N_z + 1)

# Directories
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
grandgrandparent_dir = os.path.dirname(grandparent_dir)
outputs_folder = os.path.join(grandgrandparent_dir, "Outputs", "Run1")
save_csv_location = os.path.join(grandparent_dir, "Force Analysis")
force_plots_dir = os.path.join(grandparent_dir, "Force Analysis")
os.makedirs(save_csv_location, exist_ok=True)
os.makedirs(force_plots_dir, exist_ok=True)

mixing_csv_path = os.path.join(grandparent_dir, "Segregation Analysis", "all_simulation_data_plots_mcc.csv")

# Steady state configuration
window_steady = 5
tol = 0.0075
ignore_time = 2.0
spline_s = None

def parse_simulation_folder(folder_name):
    mcc_pattern = r"sim_MCC_(?P<screw_spacing>\d+PD)_(?P<rotations>(?:[0-9.]+)?N)_(?P<rpm>\d+)"
    binary_pattern = r"sim_r0_r(?P<rAaa>[369])_(?P<screw_spacing>\d+PD)_(?P<rotations>(?:[0-9.]+)?N)_(?P<rpm>\d+)"
    
    for pattern, simtype in [(mcc_pattern, "MCC"), (binary_pattern, "Binary")]:
        match = re.match(pattern, folder_name)
        if match:
            params = match.groupdict()
            params["simtype"] = simtype
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
            elif key.lower() == "f" and arr.ndim == 2 and arr.shape[1] == 3:
                df["fx"] = arr[:, 0]
                df["fy"] = arr[:, 1]
                df["fz"] = arr[:, 2]
            else:
                df[key] = arr[:, 0] if arr.ndim > 1 else arr
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}", flush=True)
        return None

def bin_data_along_x(df, x_edges, value_column):
    bin_means = np.zeros(len(x_edges) - 1)
    for i in range(len(x_edges) - 1):
        mask = (df["x"] >= x_edges[i]) & (df["x"] < x_edges[i + 1])
        if mask.sum() > 0:
            bin_means[i] = df.loc[mask, value_column].mean()
        else:
            bin_means[i] = np.nan
    return bin_means

def analyze_axial_force_profile(simulation_time_series, x_edges):
    axial_profiles = []
    total_steps = len(simulation_time_series)
    
    for idx, entry in enumerate(simulation_time_series):
        print(f"Processing particle data: Step {idx+1} of {total_steps}", flush=True)
        df = entry["df"]
        if not all(col in df.columns for col in ["fx", "fy", "fz"]):
            continue
        df["force_mag"] = np.sqrt(df["fx"]**2 + df["fy"]**2 + df["fz"]**2)
        profile = bin_data_along_x(df, x_edges, "force_mag")
        axial_profiles.append(profile)
    
    if len(axial_profiles) == 0:
        return None
    return np.nanmean(np.stack(axial_profiles, axis=0), axis=0)

def analyze_contact_forces(sim_folder):
    contact_pattern = os.path.join(sim_folder, "pairs_*.vtk")
    contact_files = glob.glob(contact_pattern)
    if not contact_files:
        print(f"No contact files found in {sim_folder}", flush=True)
        return None

    all_normal_magnitudes = []
    all_tangential_magnitudes = []
    total_steps = len(contact_files)
    
    for idx, file in enumerate(contact_files):
        print(f"Processing contact file for statistics: Step {idx+1} of {total_steps}", flush=True)
        try:
            mesh = pv.read(file)
            if "force_normal" in mesh.cell_data and "force_tangential" in mesh.cell_data:
                fn = mesh.cell_data["force_normal"]
                ft = mesh.cell_data["force_tangential"]
                all_normal_magnitudes.extend(np.linalg.norm(fn, axis=1))
                all_tangential_magnitudes.extend(np.linalg.norm(ft, axis=1))
        except Exception as e:
            print(f"Error reading contact data from {file}: {e}", flush=True)
            continue

    if not all_normal_magnitudes or not all_tangential_magnitudes:
        print(f"No valid contact force data in {sim_folder}", flush=True)
        return None

    all_normal_magnitudes = np.array(all_normal_magnitudes)
    all_tangential_magnitudes = np.array(all_tangential_magnitudes)
    
    stats = {
        "normal_mean": np.mean(all_normal_magnitudes),
        "normal_std": np.std(all_normal_magnitudes),
        "tangential_mean": np.mean(all_tangential_magnitudes),
        "tangential_std": np.std(all_tangential_magnitudes)
    }
    return stats, all_normal_magnitudes, all_tangential_magnitudes

def get_time_from_filename(filename):
    base = os.path.basename(filename)
    m = re.search(r"pairs_(\d+)\.vtk", base)
    return m.group(1) * timestep_interval if m else None

def analyze_contact_force_heatmap(sim_folder, steady_state_time, x_edges, y_edges):
    contact_pattern = os.path.join(sim_folder, "pairs_*.vtk")
    contact_files = glob.glob(contact_pattern)
    if not contact_files:
        print(f"No contact files found in {sim_folder}", flush=True)
        return None

    heatmaps = []
    total_steps = len(contact_files)
    
    for idx, file in enumerate(contact_files):
        print(f"Processing contact file for heatmap: Step {idx+1} of {total_steps}", flush=True)
        time_val = get_time_from_filename(file)
        if time_val is None or time_val < steady_state_time:
            continue

        try:
            mesh = pv.read(file)
            centers = mesh.cell_centers().points
            
            if "force" in mesh.cell_data:
                force_vec = mesh.cell_data["force"]
                fmag = np.linalg.norm(force_vec, axis=1)
            elif "force_normal" in mesh.cell_data and "force_tangential" in mesh.cell_data:
                fn = mesh.cell_data["force_normal"]
                ft = mesh.cell_data["force_tangential"]
                fmag = np.sqrt(np.linalg.norm(fn, axis=1)**2 + np.linalg.norm(ft, axis=1)**2)
            else:
                continue

            x = centers[:, 0]
            y = centers[:, 1]
            heatmap_sum = np.zeros((len(x_edges)-1, len(y_edges)-1))
            heatmap_count = np.zeros((len(x_edges)-1, len(y_edges)-1))
            
            for xi, yi, f in zip(x, y, fmag):
                i = np.searchsorted(x_edges, xi) - 1
                j = np.searchsorted(y_edges, yi) - 1
                if 0 <= i < heatmap_sum.shape[0] and 0 <= j < heatmap_sum.shape[1]:
                    heatmap_sum[i, j] += f
                    heatmap_count[i, j] += 1
                    
            with np.errstate(invalid='ignore'):
                heatmap = np.divide(heatmap_sum, heatmap_count, out=np.zeros_like(heatmap_sum), where=heatmap_count!=0)
            heatmaps.append(heatmap)
        except Exception as e:
            print(f"Error processing contact file {file}: {e}", flush=True)
            continue

    if not heatmaps:
        print("No contact heatmap data collected in steady state.", flush=True)
        return None

    return np.nanmean(np.stack(heatmaps, axis=0), axis=0)

def get_steady_state_time(simulation_name):
    try:
        df = pd.read_csv(mixing_csv_path)
        df_sim = df[df["Simulation"] == simulation_name]
        if df_sim.empty:
            print(f"No mixing metric data found for simulation {simulation_name}", flush=True)
            return None

        steady_time = df_sim["Steady State Time (s)"].iloc[0]
        if pd.isna(steady_time):
            print(f"No steady state time found for simulation {simulation_name}", flush=True)
            return None

        print(f"Using steady state time for {simulation_name}: {steady_time:.2f} s", flush=True)
        return steady_time
    except Exception as e:
        print(f"Error loading mixing metric CSV: {e}", flush=True)
        return None

def plot_force_data(data, title, xlabel, ylabel, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data[0], data[1], marker='o', linestyle='-')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    if save_path:
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}", flush=True)
    plt.close(fig)

def plot_contact_force_histograms(all_normal, all_tangential, sim_label, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    axs[0].hist(all_normal, bins=50, color='skyblue', edgecolor='k', alpha=0.7)
    axs[0].set_title("Histogram of Normal Forces")
    axs[0].set_xlabel("Force [N]")
    axs[0].set_ylabel("Count")
    
    axs[1].hist(all_tangential, bins=50, color='salmon', edgecolor='k', alpha=0.7)
    axs[1].set_title("Histogram of Tangential Forces")
    axs[1].set_xlabel("Force [N]")
    
    fig.suptitle(f"Contact Force Distributions: {sim_label}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path)
        print(f"Contact force histogram saved to {save_path}", flush=True)
    plt.close(fig)

def plot_contact_force_heatmap(heatmap, x_edges, y_edges, sim_label, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(heatmap.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Time- and Depth-Averaged Contact Force Heatmap: {sim_label}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Contact Force Magnitude [N]")
    if save_path:
        fig.savefig(save_path)
        print(f"Contact force heatmap saved to {save_path}", flush=True)
    plt.close(fig)

def create_faceted_plots(df, plot_type, sim_type, force_plots_dir):
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f"{plot_type} - {sim_type} Simulations", fontsize=16)
    axes_flat = axes.flatten()
    
    for i, rpm in enumerate(sorted(df["RPM"].unique())):
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i]
        df_rpm = df[df["RPM"] == rpm]
        
        for spacing in sorted(df_rpm["Screw Spacing (PD)"].unique()):
            df_spacing = df_rpm[df_rpm["Screw Spacing (PD)"] == spacing]
            
            for pitch in sorted(df_spacing["Pitch (N)"].unique()):
                df_pitch = df_spacing[df_spacing["Pitch (N)"] == pitch]
                
                for sim in df_pitch["Simulation"].unique():
                    df_sim = df_pitch[df_pitch["Simulation"] == sim]
                    
                    if plot_type == "Axial Force Profile":
                        ax.plot(df_sim["Axial Position (m)"], df_sim["Avg Particle Force (N)"], 
                                label=f"{spacing}PD, {pitch}N")
                    else:  # Contact Force Heatmap
                        x_unique = sorted(df_sim["x (m)"].unique())
                        y_unique = sorted(df_sim["y (m)"].unique())
                        X, Y = np.meshgrid(x_unique, y_unique)
                        Z = np.zeros((len(x_unique), len(y_unique)))
                        
                        for idx, row in df_sim.iterrows():
                            x_idx = x_unique.index(row["x (m)"])
                            y_idx = y_unique.index(row["y (m)"])
                            Z[x_idx, y_idx] = row["Avg Contact Force (N)"]
                        
                        im = ax.imshow(Z.T, origin='lower', 
                                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                                      aspect='auto', cmap='viridis')
                        fig.colorbar(im, ax=ax, label="Average Contact Force (N)")
        
        ax.set_title(f"RPM: {rpm}")
        if plot_type == "Axial Force Profile":
            ax.set_xlabel("Axial Position (m)")
            ax.set_ylabel("Average Particle Force (N)")
            ax.grid(True)
            ax.legend()
        else:
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
    
    for i in range(len(df["RPM"].unique()), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(force_plots_dir, f"faceted_{plot_type.lower().replace(' ', '_')}_{sim_type}.png"))
    plt.close()

def main():
    simulation_results = {}
    axial_force_profiles = {}
    contact_force_stats = {}
    contact_heatmaps = {}
    all_axial_rows = []
    all_heatmap_rows = []

    for folder in os.listdir(outputs_folder):
        folder_path = os.path.join(outputs_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        sim_params = parse_simulation_folder(folder)
        if sim_params is None:
            print(f"Skipping folder '{folder}': simulation parameters not recognized.", flush=True)
            continue

        steady_time = get_steady_state_time(folder)
        if steady_time is None:
            print(f"Steady state not reached for simulation {folder}. Using last timestep value instead.", flush=True)
            steady_time = None
        
        particle_files = get_all_particle_files(folder_path)
        if not particle_files:
            print(f"No particle files found in folder '{folder}'. Skipping simulation.", flush=True)
            continue

        print(f"Processing simulation: {folder} with {len(particle_files)} particle timesteps.", flush=True)
        sim_data_file = os.path.join(save_csv_location, f"{folder}_force_data.pkl")
        simulation_time_series = []
        
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
                    "mixing_metric": np.nan
                })
            simulation_time_series.sort(key=lambda entry: entry["time"])
            with open(sim_data_file, "wb") as f:
                pickle.dump(simulation_time_series, f)
            print(f"Saved particle force data for {folder} to {sim_data_file}", flush=True)

        simulation_results[folder] = {"params": sim_params, "time_series": simulation_time_series}

        if steady_time is None:
            steady_time = simulation_time_series[-1]["time"]
            print(f"Using last timestep ({steady_time:.2f} s) as steady state time for {folder}", flush=True)

        steady_series = [entry for entry in simulation_time_series if entry["time"] >= steady_time]
        if not steady_series:
            print(f"No particle data available after steady state for {folder}. Skipping simulation.", flush=True)
            continue

        # Analysis 1: Axial Force Profile
        avg_profile = analyze_axial_force_profile(steady_series, x_edges)
        if avg_profile is not None:
            axial_force_profiles[folder] = avg_profile
            x_midpoints = (x_edges[:-1] + x_edges[1:]) / 2.0
            spacing_val = int(re.search(r"(\d+)", sim_params["screw_spacing"]).group(1))
            screw_pitch = sim_params["rotations"].strip()
            match = re.search(r"([\d.]+)N", screw_pitch)
            pitch_val = float(match.group(1)) if match else 1.0
            rpm_val = int(sim_params["rpm"])
            sim_type = sim_params["simtype"]
            
            for pos, force_val in zip(x_midpoints, avg_profile):
                all_axial_rows.append([
                    folder, sim_type, spacing_val, pitch_val, rpm_val, 
                    pos, force_val, True, steady_time
                ])
        else:
            print(f"No axial force profile generated for {folder}.", flush=True)

        # Analysis 2: Contact Force Statistics
        stats_and_data = analyze_contact_forces(folder_path)
        if stats_and_data is not None:
            stats, all_normal, all_tangential = stats_and_data
            contact_force_stats[folder] = stats
            plot_path = os.path.join(force_plots_dir, f"{folder}_contact_force_histograms.png")
            plot_contact_force_histograms(all_normal, all_tangential, folder, save_path=plot_path)
            print(f"Contact force stats for {folder}: {stats}", flush=True)
        else:
            print(f"Skipping contact force statistics for {folder} due to missing data.", flush=True)

        # Analysis 3: Contact Force Heatmap
        heatmap = analyze_contact_force_heatmap(folder_path, steady_time, x_edges, y_edges)
        if heatmap is not None:
            contact_heatmaps[folder] = heatmap
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

    # Save summary CSVs
    df_axial_forces = pd.DataFrame(all_axial_rows, 
        columns=["Simulation", "Simulation Type", "Screw Spacing (PD)", "Pitch (N)", "RPM", 
                 "Axial Position (m)", "Avg Particle Force (N)", "Steady State Reached", "Steady State Time (s)"])
    df_axial_forces.to_csv(os.path.join(save_csv_location, "axial_force_profile_data.csv"), index=False)
    print("Saved axial force profile summary CSV.", flush=True)

    contact_rows = []
    for folder, stats in contact_force_stats.items():
        contact_rows.append([folder, stats["normal_mean"], stats["normal_std"],
                           stats["tangential_mean"], stats["tangential_std"]])
    df_contact_stats = pd.DataFrame(contact_rows, 
                                    columns=["Simulation", "Normal Force Mean", "Normal Force Std", 
                                             "Tangential Force Mean", "Tangential Force Std"])
    df_contact_stats.to_csv(os.path.join(save_csv_location, "contact_force_stats.csv"), index=False)
    print("Saved contact force statistics summary CSV.", flush=True)

    df_heatmap = pd.DataFrame(all_heatmap_rows, 
        columns=["Simulation", "Simulation Type", "Screw Spacing (PD)", "Pitch (N)", "RPM", 
                 "x (m)", "y (m)", "Avg Contact Force (N)", "Steady State Reached", "Steady State Time (s)"])
    df_heatmap.to_csv(os.path.join(save_csv_location, "contact_force_heatmap_data.csv"), index=False)
    print("Saved contact force heatmap summary CSV.", flush=True)

    # Create faceted plots
    print("Creating faceted plots...", flush=True)
    for sim_type in df_axial_forces["Simulation Type"].unique():
        df_type = df_axial_forces[df_axial_forces["Simulation Type"] == sim_type]
        create_faceted_plots(df_type, "Axial Force Profile", sim_type, force_plots_dir)
        print(f"Created faceted axial force profile plot for {sim_type}", flush=True)

    for sim_type in df_heatmap["Simulation Type"].unique():
        df_type = df_heatmap[df_heatmap["Simulation Type"] == sim_type]
        create_faceted_plots(df_type, "Contact Force Heatmap", sim_type, force_plots_dir)
        print(f"Created faceted contact force heatmap plot for {sim_type}", flush=True)

    print("Force analysis complete.", flush=True)

if __name__ == "__main__":
    main()
