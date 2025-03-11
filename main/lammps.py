import numpy as np
import os
import yaml
import shutil
import matplotlib.pyplot as plt
import emcee
from functools import lru_cache
from statsmodels.tsa.stattools import acf
from matplotlib.cm import viridis

# ===== CONFIGURATION FUNCTIONS =====

def parse_input_file(input_filepath="lammps/lmp.input"):
    """Extract simulation parameters from LAMMPS input file."""
    target_vars = ['temp', 'Nstretch', 'Nequilib', 'Nrun', 'tsamp', 'tdump']
    values = {}
    
    try:
        with open(input_filepath, 'r') as file:
            for line in file:
                if line.startswith('variable'):
                    parts = line.split()
                    if parts[1] in target_vars and parts[2] == 'equal':
                        values[parts[1]] = int(float(parts[3]))
        return values
    except FileNotFoundError:
        print(f"Input file not found: {input_filepath}")
        return None


def generate_ids(values, L):
    """Generate experiment and simulation IDs from parameters."""
    values['L'] = L
    exp_id = f"t{values['temp']}_ns{values['Nstretch']}_ne{values['Nequilib']}_nr{values['Nrun']}_ts{values['tsamp']}_td{values['tdump']}"
    sim_id = f"l{L}_{exp_id}"
    return exp_id, sim_id


def get_file_paths(exp_id, sim_id, L):
    """Generate standardized file paths for a simulation."""
    base_dir = os.path.join('simulations', exp_id)
    sim_dir = os.path.join(base_dir, sim_id)
    
    return {
        'base_dir': base_dir,
        'sim_dir': sim_dir,
        'config': os.path.join(sim_dir, 'config.yaml'),
        'forces': os.path.join(sim_dir, f'lmp{L}.forces'),
        'log': os.path.join(sim_dir, f'lmp{L}.log'),
        'xyz': os.path.join(sim_dir, f'lmp{L}.xyz'),
        'force_analysis': os.path.join(sim_dir, 'force_analysis.yaml'),
        'acf_plot': os.path.join(sim_dir, 'acf_plot.png'),
        'acf_plot_halfs': os.path.join(sim_dir, 'acf_plot_halfs.png'),
        'force_time_plot': os.path.join(sim_dir, 'force_time_plot.png'),
        'force_plot': os.path.join(base_dir, 'force_plot.png')
    }


# ===== DATA MANAGEMENT =====

def store_simulation(L, input_filepath="lammps/lmp.input"):
    """Store simulation data and configuration."""
    print(f"-> Stored configuration and outputs for L={L}")
    
    # Get simulation parameters and IDs
    values = parse_input_file(input_filepath)
    if not values:
        return None, None
    
    exp_id, sim_id = generate_ids(values, L)
    paths = get_file_paths(exp_id, sim_id, L)
    
    # Create output directory
    os.makedirs(paths['sim_dir'], exist_ok=True)
    
    # Save config to YAML
    with open(paths['config'], 'w') as f:
        yaml.dump(values, f, default_flow_style=False)
    
    # Copy output files
    for src, dst in [
        (os.path.join('lammps', f'lmp{L}.xyz'), paths['xyz']),
        (os.path.join('lammps', f'lmp{L}.forces'), paths['forces']),
        (os.path.join('lammps', f'lmp{L}.log'), paths['log'])
    ]:
        try:
            shutil.copy2(src, dst)
        except FileNotFoundError:
            print(f"Warning: {src} not found")
    
    return paths['sim_dir'], values


def find_simulation_dirs(exp_id):
    """Find all simulation directories for a given experiment ID."""
    sim_dirs = []
    L_values = []
    
    base_dir = os.path.join('simulations', exp_id)
    if not os.path.exists(base_dir):
        raise ValueError(f"Experiment directory not found: {base_dir}")
    
    for dirname in os.listdir(base_dir):
        if dirname.startswith('l') and dirname[dirname.find('_')+1:] == exp_id:
            # Extract L value from directory name
            L = int(dirname[1:dirname.find('_')])
            L_values.append(L)
            sim_dirs.append((L, os.path.join(base_dir, dirname)))
    
    return sorted(sim_dirs, key=lambda x: x[0])


def get_experiment_tsamp(exp_id):
    """Extract the tsamp value from an experiment's config files."""
    sim_dirs = find_simulation_dirs(exp_id)
    
    for _, sim_dir in sim_dirs:
        config_path = os.path.join(sim_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'tsamp' in config:
                    return config['tsamp']
    
    return None


# ===== ANALYSIS FUNCTIONS =====

@lru_cache(maxsize=32)
def load_forces(force_filepath):
    """Load force data from file with caching."""
    return np.loadtxt(force_filepath)


def calculate_restoring_forces(forces):
    """Calculate restoring forces from raw force data."""
    fxh, fxt = forces[:, 0], forces[:, 3] 
    return fxh - fxt


def compute_statistics(forces):
    """Compute basic statistics for force data."""
    mean_force = np.mean(forces)
    var_force = np.var(forces)
    std_force = np.sqrt(var_force)
    return mean_force, std_force, var_force


def compute_autocorrelation(x):
    """Compute autocorrelation function for a time series."""
    
    return emcee.autocorr.function_1d(x)
    #max_lag = len(x) - 1  
    #acf_values = acf(x, nlags=max_lag, fft=True, adjusted=False)
    
    #return acf_values


def compute_tau_int(acf, tsamp, M=200):
    """Compute integrated autocorrelation time."""
    
    M = min(M, len(acf) - 1)
    
    # tau_est = 1.0 + 2.0 * np.sum(acf[1:M+1])
    tau_est = 0.5 + np.sum(acf[1:M+1])
    tau_int = tau_est * tsamp
    
    return tau_int, M


def analyze_forces(exp_id, tsamp=None, M=200, xlim_acf_anal=500, xlim_acf=1000):
    """Process force data and calculate statistics for all L values in an experiment."""
    print(f"-> Analyzing forces for experiment: {exp_id}")
    
    # If tsamp not provided, try to get from config
    if tsamp is None:
        tsamp = get_experiment_tsamp(exp_id)
        if tsamp is None:
            raise ValueError("Could not determine tsamp value. Please provide it as an argument.")
    
    # Get all simulation directories
    sim_dirs = find_simulation_dirs(exp_id)
    
    # Process all L values
    results_list = []
    
    for L, _ in sim_dirs:
        print(f"-> Analyze forces for L={L}")
        
        _, sim_id = generate_ids({'temp': exp_id.split('_')[0][1:], 
                                'Nstretch': exp_id.split('_')[1][2:],
                                'Nequilib': exp_id.split('_')[2][2:], 
                                'Nrun': exp_id.split('_')[3][2:],
                                'tsamp': exp_id.split('_')[4][2:],
                                'tdump': exp_id.split('_')[5][2:]}, L)
        
        paths = get_file_paths(exp_id, sim_id, L)
        
        # Load and process force data
        forces = load_forces(paths['forces'])
        restoring_forces = calculate_restoring_forces(forces)
            
        # Calculate statistics
        mean_force, std_force, _ = compute_statistics(restoring_forces)
            
        # Calculate autocorrelation function and integrated time
        acf = compute_autocorrelation(restoring_forces)
        tau_int, M_used = compute_tau_int(acf, tsamp, M)
            
        # Calculate effective sample size and error estimation
        N = len(restoring_forces)
        N_eff = N / (2 * tau_int)
        error_bar = np.sqrt((2 * tau_int) / N) * std_force
            
        # Create results dictionary
        results = {
            'displacement': L,
            'mean_force': float(mean_force),
            'std_force': float(std_force),
            'error_bar': float(error_bar),
            'tau_int': float(tau_int),
            'N_eff': float(N_eff),
            'mean_acf': float(np.mean(acf)),
            'M': M_used,
        }
        
        for key, value in results.items():
            if key in ["displacement"]:
                continue
            
            print(f"   - {key}: {value}")
        
        print()
            
        # Save results
        with open(paths['force_analysis'], 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Generate plots
        plot_autocorrelation_halfs(restoring_forces, xlim_acf_anal, paths['acf_plot_halfs'])
        plot_autocorrelation(acf, M_used, xlim_acf, paths['acf_plot'])
        plot_force_time_series(restoring_forces, tsamp, paths['force_time_plot'])
        
        results_list.append(results)
    
    # Create the summary plot for all L values
    plot_mean_forces(exp_id)
    
    return results_list


# ===== VISUALIZATION FUNCTIONS =====

def format_experiment_title(exp_id):
    """Format experiment ID into a readable plot title."""
    if not exp_id.startswith('t'):
        return exp_id
        
    try:
        # Parse the experiment ID
        parts = exp_id.split('_')
        temp = int(parts[0][1:])
        ns = int(parts[1][2:])
        ne = int(parts[2][2:])
        nr = int(parts[3][2:])
        ts = int(parts[4][2:])
        
        # Handle special case
        if exp_id == "t700_ns10000_ne10000_nr1000000_ts20_td1000":
            return "700K - Stretch 1e4 - Equil 1e4 - Prod 1e6 - Tsamp 20"
        
        # Format scientific notation
        def format_sci(n):
            """Format number in scientific notation.
            Examples: 10000 → 1e4, 50000 → 5e4, 500000 → 5e5"""
            n_str = str(int(n))  # Convert to int first to handle potential floats
            
            if n >= 1000:
                first_digit = n_str[0]
                power = len(n_str) - 1
                return f"{first_digit}e{power}"
                
            return n_str
        
        return f"{temp}K - Stretch {format_sci(ns)} - Equil {format_sci(ne)} - Prod {format_sci(nr)} - Tsamp {ts}"
    except:
        return exp_id


def plot_force_time_series(forces, tsamp, output_path):
    """Generate and save raw force vs time plot without statistics."""
    # Create time axis in simulation time units
    time_steps = np.arange(len(forces)) * tsamp
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot only the raw force data with viridis color
    ax.plot(time_steps, forces, color=viridis(0.3), linewidth=1)
    
    ax.set_xlabel('Simulation Time', fontsize=12)
    ax.set_ylabel('Restoring Force', fontsize=12)
    ax.set_title('Force vs Time', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_autocorrelation(acf, M, xlim, output_path):
    """Generate and save autocorrelation function plot with M cutoff indicator."""
    
    beg_acf = acf
    
    t = np.arange(len(beg_acf))
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot autocorrelation function with viridis color
    ax.plot(t, beg_acf, color=viridis(0.3), label='Autocorrelation')
    
    # Add vertical line for M cutoff with viridis color
    if M < len(beg_acf):
        ax.axvline(x=M, color=viridis(0.8), linestyle='--', label=f'Cutoff M={M}')
        
        # Add a point at the cutoff value with viridis color
        ax.plot(M, beg_acf[M], 'o', color=viridis(0.8), markersize=6)
    
    ax.set_xlabel(rf'Time lag $\tau$', fontsize=12)
    ax.set_ylabel(rf'Autocorrelation $\rho(\tau)$', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim([0, xlim])
    ax.set_ylim([-0.5, 1.1])
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_autocorrelation_halfs(x, xlim, output_path):
    """Generate and save autocorrelation function plot comparing first and second halves of data on a single plot."""
    
    # Split the time series into two halves
    half_point = len(x) // 2
    first_half = x[:half_point]
    second_half = x[half_point:]
    
    # Compute autocorrelation for both halves
    acf_first = compute_autocorrelation(first_half)
    acf_second = compute_autocorrelation(second_half)
    
    # Create a single plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot autocorrelation for both halves with viridis colors
    t1 = np.arange(len(acf_first))
    t2 = np.arange(len(acf_second))
    
    ax.plot(t1, acf_first, color=viridis(0.3), label='First Half')
    ax.plot(t2, acf_second, color=viridis(0.7), label='Second Half')
    
    ax.set_xlabel(rf'Time lag $\tau$', fontsize=12)
    ax.set_ylabel(rf'Autocorrelation $\rho(\tau)$', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_ylim([-0.5, 1.1])
    ax.set_xlim([0, xlim])
    ax.set_title('Comparison of Autocorrelation Functions')
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mean_forces(exp_id):
    """Generate and save mean forces vs displacement plot."""
    print(f"->Generating mean forces plot for experiment: {exp_id}\n")
    sim_dirs = find_simulation_dirs(exp_id)
    
    if not sim_dirs:
        raise ValueError(f"No data found for experiment: {exp_id}")
    
    # Collect data from all simulations
    data = []
    for L, sim_dir in sim_dirs:
        yaml_path = os.path.join(sim_dir, 'force_analysis.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                results = yaml.safe_load(f)
                data.append((
                    results['displacement'],
                    results['mean_force'],
                    results['error_bar']
                ))
    
    if not data:
        raise ValueError(f"No force analysis data found for experiment: {exp_id}")
    
    # Sort by displacement
    data.sort(key=lambda x: x[0])
    displacements, mean_forces, error_bars = zip(*data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 7))
    
    # Use viridis colors for errorbar plot
    ax.errorbar(
        displacements, mean_forces, yerr=error_bars, 
        fmt='o-', capsize=5, capthick=1.5, 
        elinewidth=1.5, markersize=8,
        color=viridis(0.5), ecolor=viridis(0.7), 
        mfc=viridis(0.3), mec=viridis(0.7),
        label='Mean Force'
    )
    
    ax.set_xlabel('Displacement (L)', fontsize=12)
    ax.set_ylabel('Mean Force', fontsize=12)
    ax.set_title(f'Force vs Displacement\n{format_experiment_title(exp_id)}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    paths = get_file_paths(exp_id, "", 0)  # L=0 not used here
    plt.savefig(paths['force_plot'], dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig


# ===== MAIN EXECUTION =====

def process_experiment(Ls, save_results=True, process_forces=True, exp_id=None, tsamp=None, M=None, xlim_acf_anal=500, xlim_acf=1000):
    """Process an entire experiment with multiple displacements."""
    
    print(f"Processing experiment: {exp_id}")
    
    values = None
    
    for L in Ls:
        
        print(f"\nProcessing L={L}")
        
        if save_results:
            # Store simulation data
            output_dir, values = store_simulation(L)
            
            # Get experiment ID from values if not provided
            if exp_id is None and values is not None:
                exp_id, _ = generate_ids(values, L)
            
            # Update tsamp if available
            if values and 'tsamp' in values:
                tsamp = values['tsamp']
    
    if process_forces and exp_id:
        # Analyze forces for all L values
        analyze_forces(exp_id, tsamp, M, xlim_acf_anal, xlim_acf)
        
    return exp_id


def analysis_300K():
    # tsamp 10 
    analyze_forces("t300_ns100000_ne50000_nr1000000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=1000) # at L=180 messy
    analyze_forces("t300_ns100000_ne50000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=1000) # at L=150 messy
    analyze_forces("t300_ns100000_ne100000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=1000) # at L=140 messy
    analyze_forces("t300_ns100000_ne1000000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=1000) # at L=140 messy
    
    # tsamp 20 
    analyze_forces("t300_ns100000_ne50000_nr1000000_ts20_td1000", tsamp=20, M=15, xlim_acf_anal=100, xlim_acf=1000) # at L=180 messy
    analyze_forces("t300_ns100000_ne50000_nr1500000_ts20_td1000", tsamp=20, M=15, xlim_acf_anal=100, xlim_acf=1000) # at L=150 messy
    
    
def analysis_700K():
    # tsamp 10 
    analyze_forces("t700_ns10000_ne5000_nr1000000_ts10_td1000", tsamp=10, M=15, xlim_acf_anal=200, xlim_acf=1000) 
    
    # tsamp 20
    analyze_forces("t700_ns10000_ne10000_nr1000000_ts20_td1000", tsamp=20, M=30, xlim_acf_anal=100, xlim_acf=1000) 
    analyze_forces("t700_ns10000_ne50000_nr1000000_ts20_td1000", tsamp=20, M=30, xlim_acf_anal=100, xlim_acf=1000) 
    analyze_forces("t700_ns100000_ne10000_nr1000000_ts20_td1000", tsamp=20, M=30, xlim_acf_anal=100, xlim_acf=1000) 
    analyze_forces("t700_ns100000_ne50000_nr1000000_ts20_td1000", tsamp=20, M=30, xlim_acf_anal=100, xlim_acf=1000) 
    
    
if __name__ == "__main__":
    analysis_300K()