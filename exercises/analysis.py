import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from statsmodels.tsa.stattools import acf

# GET DATA #
def get_file_paths(simulation_path):
    """Get standard file paths following simulation naming convention."""

    # Create all directories in the path
    os.makedirs(simulation_path, exist_ok=True)
    os.makedirs(os.path.join(simulation_path, 'plots'), exist_ok=True)

    return {
        'xyz': os.path.join(simulation_path, "positions.xyz"),
        'distances': os.path.join(simulation_path, "distances.csv"),
        'stretches': os.path.join(simulation_path, "stretches.csv"),
        'plots': os.path.join(simulation_path, "plots"),
    }

# HELPER FUNCTIONS #
def get_direction_vector(positions, pos1, pos2):
    """Calculate unit vector between two atoms."""
    direction = positions[pos2] - positions[pos1]
    return direction / np.linalg.norm(direction)

def autocorrelation(x):
    """Calculate autocorrelation of a signal."""
    return acf(x, nlags=100, fft=True)

# BASIC ANALYSIS #
def analyze_distances(simulation_path):
    """Analyze distance evolution and autocorrelation."""
    paths = get_file_paths(simulation_path)
    distances = pd.read_csv(paths['distances'])

    # Distance evolution plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(distances['time'], distances['distance'])
    plt.xlabel('Time (fs)')
    plt.ylabel('Distance (Å)')
    plt.title('Distance Evolution')
    plt.grid(True)

    plot_path = os.path.join(paths['plots'], 'distance_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Autocorrelation plot
    acf_values = autocorrelation(distances['distance'].values)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(acf_values, color='r')
    plt.xlabel('Lag Steps')
    plt.ylabel('ACF')
    plt.title('Distance Autocorrelation')
    plt.grid(True)

    acf_path = os.path.join(paths['plots'], 'distance_autocorrelation.png')
    plt.savefig(acf_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'distance_plot': plot_path,
        'acf_plot': acf_path,
        'data': distances,
        'acf': acf_values
    }

def analyze_stretches(simulation_path):
    """Analyze and plot stretches evolution."""
    paths = get_file_paths(simulation_path)

    if not os.path.exists(paths['stretches']):
        return None

    stretches = pd.read_csv(paths['stretches'])

    fig = plt.figure(figsize=(10, 6))
    plt.plot(stretches['time'], stretches['cumulative_stretch'])
    plt.xlabel('Time (fs)')
    plt.ylabel('Cumulative Stretch (Å)')
    plt.title('Applied Stretches')
    plt.grid(True)

    # Save figure
    plot_path = os.path.join(paths['plots'], 'stretches_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {'plot': plot_path, 'data': stretches}

# FORCE ANALYSIS #
def calculate_force(simulation_path, pos1, pos2):
    paths = get_file_paths(simulation_path)

    if not os.path.exists(paths['xyz']):
        return None

    forces = []

    for frame in read(paths['xyz'], index=":"):
        direction_vector = get_direction_vector(frame.get_positions(), pos1, pos2)
        force_difference = frame.get_forces()[pos1,:] - frame.get_forces()[pos2,:]
        force_projection = np.dot(force_difference, direction_vector)
        forces.append(force_projection)

    return forces

def analyze_forces_complete(simulation_path, pos1, pos2, equilibration_steps=5000):
    paths = get_file_paths(simulation_path)

    # Calculate forces and distances for each frame
    forces = []
    distances = []

    for frame in read(paths['xyz'], index=":"):
        # Get positions and calculate distance
        positions = frame.get_positions()
        distance = np.linalg.norm(positions[pos2] - positions[pos1])
        distances.append(distance)

        # Calculate force
        direction_vector = get_direction_vector(positions, pos1, pos2)
        force_difference = frame.get_forces()[pos1,:] - frame.get_forces()[pos2,:]
        force_projection = np.dot(force_difference, direction_vector)
        forces.append(force_projection)

    # Create DataFrame
    data = pd.DataFrame({
        'time': np.arange(len(forces)),
        'force': forces,
        'distance': distances
    })

    # Skip equilibration period
    data = data[equilibration_steps:]

    ### PLOT Force evolution ###
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['force'])
    plt.xlabel('Time Steps')
    plt.ylabel('Restoring Force (eV/Å)')
    plt.title('Restoring Force Evolution')
    plt.grid(True)

    force_plot_path = os.path.join(paths['plots'], 'force_evolution.png')
    plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    ### PLOT Force autocorrelation ###
    acf_values = autocorrelation(data['force'].values)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(acf_values, color='r')
    plt.xlabel('Lag Steps')
    plt.ylabel('ACF')
    plt.title('Force Autocorrelation')
    plt.grid(True)

    acf_path = os.path.join(paths['plots'], 'force_autocorrelation.png')
    plt.savefig(acf_path, dpi=300, bbox_inches='tight')
    plt.close()

    ### PLOT Force vs Displacement ###
    correlation_time = np.where(acf_values < 1/np.e)[0][0] # Find time lag where autocorrelation below 1/e (characteristic time over which measurements remain correlated)

    total_measurements = len(data)
    independent_measurements = total_measurements / correlation_time # E.g. if 1000 measurements with correlation time of 10, only 100 independent measurements
    error = np.std(data['force']) / np.sqrt(independent_measurements) # Standard error formula modified for correlated data

    fig = plt.figure(figsize=(10, 6))
    plt.errorbar(data['distance'], data['force'], yerr=error, fmt='o', alpha=0.5, capsize=5, capthick=1, elinewidth=1)
    plt.xlabel('End-to-End Distance (Å)')
    plt.ylabel('Restoring Force (eV/Å)')
    plt.title('Force vs Displacement')
    plt.grid(True)

    force_vs_disp_path = os.path.join(paths['plots'], 'force_vs_displacement.png')
    plt.savefig(force_vs_disp_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'force_evolution_plot': force_plot_path,
        'force_acf_plot': acf_path,
        'force_vs_displacement_plot': force_vs_disp_path,
        'data': data,
        'mean_force': np.mean(data['force']),
        'mean_distance': np.mean(data['distance']),
        'error': error,
        'correlation_time': correlation_time,
        'acf': acf_values
    }
