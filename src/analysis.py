import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from scipy.signal import correlate
import seaborn as sns
import os

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

def analyze_distances(simulation_path):
    """Analyze and plot distance evolution."""
    paths = get_file_paths(simulation_path)
    distances = pd.read_csv(paths['distances'])
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(distances['time'], distances['distance'])
    plt.xlabel('Time (fs)')
    plt.ylabel('Distance (Å)')
    plt.title('Distance Evolution')
    plt.grid(True)
    
    # Save figure
    plot_path = os.path.join(paths['plots'], 'distance_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'plot': plot_path, 'data': distances}

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

def get_direction_vector(positions, pos1, pos2):
    """Calculate unit vector between two atoms."""
    direction = positions[pos2] - positions[pos1]
    return direction / np.linalg.norm(direction)

def calculate_force_component(forces, pos1, pos2, direction=None, positions=None):
    """
    Calculate force component along a direction.
    If direction is not provided, calculates it from positions.
    """
    if direction is None:
        if positions is None:
            raise ValueError("Either direction or positions must be provided")
        direction = get_direction_vector(positions, pos1, pos2)
    
    force_diff = forces[pos2] - forces[pos1]
    return np.dot(force_diff, direction)

def analyze_forces(simulation_path, pos1, pos2, direction=None,
                  relaxation_steps=1000, stretches_data=None):
    """Analyze forces from trajectory file."""
    paths = get_file_paths(simulation_path)
    trajectory = list(read(paths['xyz'], ':'))
    forces = []
    times = []
    distances = []
    
    last_stretch_time = -float('inf')
    
    for i, atoms in enumerate(trajectory):
        current_time = i * relaxation_steps  # Approximate time
        
        # Skip relaxation period after stretches if stretch data available
        if stretches_data is not None:
            if any(abs(current_time - st) < relaxation_steps 
                  for st in stretches_data['time']):
                last_stretch_time = current_time
                continue
        
        if current_time - last_stretch_time > relaxation_steps:
            # Calculate direction if not provided
            if direction is None:
                dir_vec = get_direction_vector(atoms.get_positions(), pos1, pos2)
            else:
                dir_vec = direction
                
            force = calculate_force_component(atoms.get_forces(), pos1, pos2,
                                           direction=dir_vec)
            distance = np.linalg.norm(
                atoms.get_positions()[pos2] - atoms.get_positions()[pos1]
            )
            
            forces.append(force)
            times.append(current_time)
            distances.append(distance)
    
    forces = np.array(forces)
    times = np.array(times)
    distances = np.array(distances)
    
    # Generate plots
    correlation_plot = plot_force_correlation(forces, times, paths['plots'])
    force_distance_plot = plot_force_distance(forces, distances, paths['plots'])
    
    return {
        'forces': forces,
        'times': times,
        'distances': distances,
        'stats': calculate_force_statistics(forces, times),
        'force_distance_plot': force_distance_plot,
        'correlation_plot': correlation_plot
    }

def autocorrelation(x):
    """Calculate autocorrelation of a signal."""
    x = x - np.mean(x)
    result = correlate(x, x, mode='full') / len(x)
    return result[len(result)//2:]

def calculate_force_statistics(forces, times, max_correlation_time=1000):
    """Calculate force statistics including error bars."""
    ac = autocorrelation(forces)
    correlation_time = np.sum(ac[:max_correlation_time]) / ac[0]
    
    # Calculate error using block averaging
    block_size = max(int(correlation_time), 1)
    n_blocks = len(forces) // block_size
    
    if n_blocks < 2:
        block_size = len(forces) // 2
        n_blocks = 2
    
    block_means = np.array([
        np.mean(forces[i*block_size:(i+1)*block_size])
        for i in range(n_blocks)
    ])
    
    mean_force = np.mean(block_means)
    error = np.std(block_means) / np.sqrt(n_blocks - 1)
    
    return {
        'mean_force': mean_force,
        'error': error,
        'correlation_time': correlation_time,
        'autocorrelation': ac
    }

def plot_force_correlation(forces, times, output_dir):
    """Plot force autocorrelation function."""
    ac = autocorrelation(forces)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(times[:len(ac)] - times[0], ac)
    plt.xlabel('Time lag (fs)')
    plt.ylabel('Force Autocorrelation')
    plt.title('Force Autocorrelation Function')
    plt.grid(True)
    
    # Save figure
    plot_path = os.path.join(output_dir, 'force_correlation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_force_distance(forces, distances, output_dir):
    """Plot force vs distance to analyze entropic/enthalpic regimes."""
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(distances, forces, alpha=0.5)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Force (eV/Å)')
    plt.title('Force-Distance Relationship')
    plt.grid(True)
    
    # Fit polynomial to show nonlinearity
    z = np.polyfit(distances, forces, 3)
    p = np.poly1d(z)
    x_fit = np.linspace(min(distances), max(distances), 100)
    plt.plot(x_fit, p(x_fit), 'r--', alpha=0.8)
    
    # Save figure
    plot_path = os.path.join(output_dir, 'force_distance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def analyze_simulation(simulation_path, pos1, pos2):
    """Main analysis function to process all simulation data."""
    # Analyze basic evolution
    distances_results = analyze_distances(simulation_path)
    stretches_results = analyze_stretches(simulation_path)
    
    # Analyze forces using stretches data if available
    stretches_data = stretches_results['data'] if stretches_results else None
    force_results = analyze_forces(
        simulation_path,
        pos1=pos1,
        pos2=pos2,
        stretches_data=stretches_data
    )
    
    print(f"Mean force: {force_results['stats']['mean_force']:.3f} ± "
          f"{force_results['stats']['error']:.3f} eV/Å")
    print(f"Correlation time: {force_results['stats']['correlation_time']:.1f} steps")
    
    return {
        'distances': distances_results,
        'stretches': stretches_results,
        'forces': force_results
    }

if __name__ == "__main__":
    # Example usage
    results = analyze_simulation(
        simulation_path="path/to/simulation/output",
        pos1=0,
        pos2=1
    )