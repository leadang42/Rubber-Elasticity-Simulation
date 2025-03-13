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


def generate_ids(params, L):
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
        'energy_analysis': os.path.join(sim_dir, 'energy_analysis.yaml'), 
        'acf_plot': os.path.join(sim_dir, 'acf_plot.png'),
        'acf_plot_halfs': os.path.join(sim_dir, 'acf_plot_halfs.png'),
        'force_time_plot': os.path.join(sim_dir, 'force_time_plot.png'),
        'energy_time_plot': os.path.join(sim_dir, 'energy_time_plot.png'),  
        'force_plot': os.path.join(base_dir, 'force_plot.png'),
        'energy_plot': os.path.join(base_dir, 'energy_plot.png'),  
        'combined_plot': os.path.join(base_dir, 'combined_plot.png') 
    }


# ===== DATA MANAGEMENT =====

def store_simulation(L, input_filepath="lammps/lmp.input"):
    """Store simulation data and configuration."""
    
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
    
    print(f"-> Stored configuration and outputs for L={L}\n")
    
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


@lru_cache(maxsize=32)
def load_log(log_filepath):
    """Load log data from file with caching."""
    data = []
    in_thermo_section = False
    found_header = False
    
    with open(log_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if "Step" in line and "Temp" in line and "PotEng" in line:
                found_header = True
                in_thermo_section = True
                continue
            
            if in_thermo_section and found_header:
                if line.startswith("Loop time") or not line:
                    in_thermo_section = False
                    continue
                
                values = line.split()
                if len(values) >= 3:
                    step = float(values[0])
                    temp = float(values[1])
                    pe = float(values[2])
                    data.append([step, temp, pe])
    
    return np.array(data) if data else np.empty((0, 3))


def calculate_internal_energy(forces):
    """Calculate internal energy from force and position data.
    
    Force file structure:
    fxh, fyh, fzh, fxt, fyt, fzt, xh, xt
    """
    # Extract components from the forces array
    fxh = forces[:, 0]
    fyh = forces[:, 1]
    fzh = forces[:, 2]
    fxt = forces[:, 3]
    fyt = forces[:, 4]
    fzt = forces[:, 5]
    xh = forces[:, 6]
    xt = forces[:, 7]
    
    # Calculate displacement
    dx = xt - xh
    
    # Calculate net force components
    fx_net = fxh - fxt
    fy_net = fyh - fyt
    fz_net = fzh - fzt
    
    # Calculate internal energy (work done by forces)
    # U = -F·dr (assuming only x-component is relevant for simplicity)
    
    internal_energy = -(fx_net * dx)
    
    return internal_energy


def compute_statistics(data):
    """Compute basic statistics for any data series."""
    mean_value = np.mean(data)
    var_value = np.var(data)
    std_value = np.sqrt(var_value)
    return mean_value, std_value, var_value


def compute_autocorrelation(x):
    """Compute autocorrelation function for a time series."""
    
    max_lag = len(x) - 1  
    acf_values = acf(x, nlags=max_lag, fft=True, adjusted=False)
    
    return acf_values


def compute_tau_int(acf, tsamp, M):
    """Compute integrated autocorrelation time."""
    
    M = min(M, len(acf) - 1)
    
    tau_est = 0.5 + np.sum(acf[1:M+1])
    tau_int = tau_est * tsamp
    
    return tau_int, M


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
    
    fig, ax = plt.subplots(figsize=(7, 2))
    
    # Plot only the raw force data 
    ax.plot(time_steps, forces, color=viridis(0.3), linewidth=1)
    
    ax.set_xlabel('Simulation Time', fontsize=12)
    ax.set_ylabel('Restoring Force', fontsize=12)
    ax.set_xlim([0, time_steps[-1]])
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_energy_time_series(energy, tsamp, output_path):
    """Generate and save raw energy vs time plot without statistics."""
    # Create time axis in simulation time units
    time_steps = np.arange(len(energy)) * tsamp
    
    fig, ax = plt.subplots(figsize=(7, 2))
    
    # Plot only the raw energy data 
    ax.plot(time_steps, energy, color=viridis(0.6), linewidth=1)
    
    ax.set_xlabel('Simulation Time', fontsize=12)
    ax.set_ylabel('Internal Energy', fontsize=12)
    ax.set_xlim([0, time_steps[-1]])
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_autocorrelation(acf, M, xlim, output_path):
    """Generate and save autocorrelation function plot with M cutoff indicator."""
    
    beg_acf = acf
    
    t = np.arange(len(beg_acf))
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot autocorrelation function 
    ax.plot(t, beg_acf, color=viridis(0.2), label='Autocorrelation')
    
    # Add vertical line for M cutoff 
    #if M < len(beg_acf):
    #    ax.axvline(x=M, color=viridis(0.8), linestyle='--', label=f'Cutoff M={M}')
    #    ax.plot(M, beg_acf[M], 'o', color=viridis(0.8), markersize=6)
    
    ax.set_xlabel(rf'Time lag $\tau$', fontsize=12)
    ax.set_ylabel(rf'Autocorrelation $\rho(\tau)$', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim([0, xlim])
    ax.set_ylim([-0.5, 0.5])
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_autocorrelation_halfs(x, M, xlim, output_path):
    """Generate and save autocorrelation function plot comparing first and second halves of data on a single plot."""
    
    # Split the time series into two halves
    half_point = len(x) // 2
    first_half = x[:half_point]
    second_half = x[half_point:]
    
    # Compute autocorrelation for both halves
    acf_first = compute_autocorrelation(first_half)
    acf_second = compute_autocorrelation(second_half)
    
    # Create a single plot
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Plot autocorrelation for both halves s
    t1 = np.arange(len(acf_first))
    t2 = np.arange(len(acf_second))
    
    ax.plot(t1, acf_first, color=viridis(0.2), label='First Half')
    ax.plot(t2, acf_second, color=viridis(0.6), label='Second Half', alpha=0.8)
    
    #if M < len(acf_first):
    #    ax.axvline(x=M, color=viridis(0.8), linestyle='--', label=f'Cutoff M={M}')
    #   ax.plot(M, acf_first[M], 'o', color=viridis(0.8), markersize=6)
        
    ax.set_xlabel(rf'Time lag $\tau$', fontsize=12)
    ax.set_ylabel(rf'Autocorrelation $\rho(\tau)$', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_ylim([-0.5, 1])
    ax.set_xlim([0, xlim])
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mean_forces(exp_id):
    """Generate and save mean forces vs displacement plot."""
    print(f"-> Generating mean forces plot for experiment: {exp_id}\n")
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
    fig, ax = plt.subplots(figsize=(7, 6))
    
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
    ax.set_ylabel('Mean Restoring Force', fontsize=12)
    ax.set_title(f'Force vs Displacement\n{format_experiment_title(exp_id)}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    paths = get_file_paths(exp_id, "", 0)  # L=0 not used here
    plt.savefig(paths['force_plot'], dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig


def plot_mean_energy(exp_id):
    """Generate and save mean internal energy vs displacement plot."""
    print(f"-> Generating mean internal energy plot for experiment: {exp_id}\n")
    sim_dirs = find_simulation_dirs(exp_id)
    
    if not sim_dirs:
        raise ValueError(f"No data found for experiment: {exp_id}")
    
    # Collect data from all simulations
    data = []
    for L, sim_dir in sim_dirs:
        yaml_path = os.path.join(sim_dir, 'energy_analysis.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                results = yaml.safe_load(f)
                data.append((
                    results['displacement'],
                    results['mean_energy'],
                    results['error_bar']
                ))
    
    if not data:
        raise ValueError(f"No energy analysis data found for experiment: {exp_id}")
    
    # Sort by displacement
    data.sort(key=lambda x: x[0])
    displacements, mean_energies, error_bars = zip(*data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Use viridis colors for errorbar plot (using a different shade than the force plot)
    ax.errorbar(
        displacements, mean_energies, yerr=error_bars, 
        fmt='o-', capsize=5, capthick=1.5, 
        elinewidth=1.5, markersize=8,
        color=viridis(0.8), ecolor=viridis(0.9), 
        mfc=viridis(0.6), mec=viridis(0.9),
        label='Mean Internal Energy'
    )
    
    ax.set_xlabel('Displacement (L)', fontsize=12)
    ax.set_ylabel('Mean Internal Energy', fontsize=12)
    ax.set_title(f'Internal Energy vs Displacement\n{format_experiment_title(exp_id)}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    paths = get_file_paths(exp_id, "", 0)  # L=0 not used here
    plt.savefig(paths['energy_plot'], dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig


def plot_combined_force_energy(exp_id):
    """Generate and save a combined plot with both mean force and energy vs displacement."""
    print(f"-> Generating combined force and energy plot for experiment: {exp_id}\n")
    sim_dirs = find_simulation_dirs(exp_id)
    
    if not sim_dirs:
        raise ValueError(f"No data found for experiment: {exp_id}")
    
    # Collect force data
    force_data = []
    for L, sim_dir in sim_dirs:
        yaml_path = os.path.join(sim_dir, 'force_analysis.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                results = yaml.safe_load(f)
                force_data.append((
                    results['displacement'],
                    results['mean_force'],
                    results['error_bar']
                ))
    
    # Collect energy data
    energy_data = []
    for L, sim_dir in sim_dirs:
        yaml_path = os.path.join(sim_dir, 'energy_analysis.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                results = yaml.safe_load(f)
                energy_data.append((
                    results['displacement'],
                    results['mean_energy'],
                    results['error_bar']
                ))
    
    if not force_data or not energy_data:
        raise ValueError(f"No complete analysis data found for experiment: {exp_id}")
    
    # Sort by displacement
    force_data.sort(key=lambda x: x[0])
    energy_data.sort(key=lambda x: x[0])
    
    # Unpack data
    force_displacements, mean_forces, force_error_bars = zip(*force_data)
    energy_displacements, mean_energies, energy_error_bars = zip(*energy_data)
    
    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    # Force plot (left y-axis)
    color1 = viridis(0.3)
    ax1.errorbar(
        force_displacements, mean_forces, yerr=force_error_bars, 
        fmt='o-', capsize=5, capthick=1.5, 
        elinewidth=1.5, markersize=8,
        color=color1, ecolor=viridis(0.4), 
        mfc=viridis(0.2), mec=viridis(0.4),
        label='Mean Restoring Force'
    )
    
    ax1.set_xlabel('Displacement (L)', fontsize=12)
    ax1.set_ylabel('Mean Restoring Force', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Energy plot (right y-axis)
    ax2 = ax1.twinx()
    color2 = viridis(0.8)
    ax2.errorbar(
        energy_displacements, mean_energies, yerr=energy_error_bars, 
        fmt='s--', capsize=5, capthick=1.5, 
        elinewidth=1.5, markersize=8,
        color=color2, ecolor=viridis(0.9), 
        mfc=viridis(0.7), mec=viridis(0.9),
        label='Mean Internal Energy'
    )
    
    ax2.set_ylabel('Mean Internal Energy', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legends
    plt.title(f'Force and Internal Energy vs Displacement\n{format_experiment_title(exp_id)}', fontsize=14)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    paths = get_file_paths(exp_id, "", 0)  # L=0 not used here
    plt.savefig(paths['combined_plot'], dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return fig


# ===== MAIN EXECUTION =====

def store_exp(Ls):
    """Process an entire experiment with multiple displacements."""
    
    values = None
    
    for L in Ls:
        
        print(f"\nProcessing L={L}")
        output_dir, values = store_simulation(L)
        
        if values is not None:
            exp_id, _ = generate_ids(values, L)


def analyze_exp(exp_id, tsamp, M, xlim_acf_anal, xlim_acf):
    """Process force data and calculate statistics for all L values in an experiment."""

    sim_dirs = find_simulation_dirs(exp_id)
    
    for L, _ in sim_dirs:
        
        params = {
            'temp': exp_id.split('_')[0][1:], 
            'Nstretch': exp_id.split('_')[1][2:], 
            'Nequilib': exp_id.split('_')[2][2:], 
            'Nrun': exp_id.split('_')[3][2:], 
            'tsamp': exp_id.split('_')[4][2:], 
            'tdump': exp_id.split('_')[5][2:], 
        }
        
        _, sim_id = generate_ids(params, L)
        paths = get_file_paths(exp_id, sim_id, L)
        forces = load_forces(paths['forces'])
        
        # Process restoring forces
        restoring_forces = calculate_restoring_forces(forces)
        
        mean_force, std_force, _ = compute_statistics(restoring_forces)
        acf_force = compute_autocorrelation(restoring_forces)
        tau_int_force, _ = compute_tau_int(acf_force, tsamp, M)
        N = len(restoring_forces)
        N_eff_force = N / (2 * tau_int_force)
        error_bar_force = np.sqrt((2 * tau_int_force) / N) * std_force
        
        force_results = {
            'displacement': L,
            'mean_force': float(mean_force),
            'std_force': float(std_force),
            'error_bar': float(error_bar_force),
            'tau_int': float(tau_int_force),
            'N_eff': float(N_eff_force),
            'mean_acf': float(np.mean(acf_force)),
            'M': M,
        }
        
        with open(paths['force_analysis'], 'w') as f:
            yaml.dump(force_results, f, default_flow_style=False)
        
        print(f"-> Force Analysis for {L}:")
        for key, value in force_results.items():
            if key in ["displacement"]:
                continue
            print(f"   - {key}: {value}")
            
        
        # Process energy data 
        run_start = (params['Nstretch'] + params['Nequilib']) / params['tsamp']
        
        pot_energy = load_log(paths['log'])[:, 2]
        
        internal_energy = calculate_internal_energy(forces)
        mean_energy, std_energy, _ = compute_statistics(internal_energy)
        acf_energy = compute_autocorrelation(internal_energy)
        tau_int_energy, _ = compute_tau_int(acf_energy, tsamp, M)
        N_eff_energy = N / (2 * tau_int_energy)
        error_bar_energy = np.sqrt((2 * tau_int_energy) / N) * std_energy
        
        energy_results = {
            'displacement': L,
            'mean_energy': float(mean_energy),
            'std_energy': float(std_energy),
            'error_bar': float(error_bar_energy),
            'tau_int': float(tau_int_energy),
            'N_eff': float(N_eff_energy),
            'mean_acf': float(np.mean(acf_energy)),
            'M': M,
        }
        
        with open(paths['energy_analysis'], 'w') as f:
            yaml.dump(energy_results, f, default_flow_style=False)
            
        # Print energy results
        print(f"-> Energy Analysis for {L}:")
        for key, value in energy_results.items():
            if key in ["displacement"]:
                continue
            print(f"   - {key}: {value}")
        
        print()
        
        # Generate plots for one simulation
        plot_autocorrelation(acf_force, M, xlim_acf, paths['acf_plot'])
        plot_autocorrelation_halfs(restoring_forces, M, xlim_acf_anal, paths['acf_plot_halfs'])
        plot_force_time_series(restoring_forces, tsamp, paths['force_time_plot'])
        plot_energy_time_series(internal_energy, tsamp, paths['energy_time_plot'])
    
    # Create the summary plots for all L values
    plot_mean_forces(exp_id)
    plot_mean_energy(exp_id)
    plot_combined_force_energy(exp_id)


def analysis_300K():
    # tsamp 10 
    analyze_exp("t300_ns100000_ne50000_nr1000000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=500) # at L=180 messy
    analyze_exp("t300_ns100000_ne50000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=500) # at L=150 messy
    # analyze_exp("t300_ns100000_ne100000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=1000) # at L=140 messy
    analyze_exp("t300_ns100000_ne1000000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=500) # at L=140 messy
    
    # tsamp 20 
    analyze_exp("t300_ns100000_ne50000_nr1000000_ts20_td1000", tsamp=20, M=15, xlim_acf_anal=200, xlim_acf=500) # at L=180 messy
    analyze_exp("t300_ns100000_ne50000_nr1500000_ts20_td1000", tsamp=20, M=15, xlim_acf_anal=200, xlim_acf=500) # at L=150 messy
    
    # tsamp 50
    analyze_exp("t300_ns100000_ne100000_nr1500000_ts50_td1000", tsamp=50, M=100, xlim_acf_anal=200, xlim_acf=500) # at L=150 messy
    
    
def analysis_700K():
    # tsamp 10 
    analyze_exp("t700_ns10000_ne5000_nr1000000_ts10_td1000", tsamp=10, M=90, xlim_acf_anal=200, xlim_acf=500) 
    
    # tsamp 20
    analyze_exp("t700_ns10000_ne10000_nr1000000_ts20_td1000", tsamp=20, M=45, xlim_acf_anal=100, xlim_acf=500) # at L=160,180,190 messy
    analyze_exp("t700_ns10000_ne50000_nr1000000_ts20_td1000", tsamp=20, M=45, xlim_acf_anal=100, xlim_acf=500) 
    analyze_exp("t700_ns100000_ne10000_nr1000000_ts20_td1000", tsamp=20, M=45, xlim_acf_anal=100, xlim_acf=500) 
    analyze_exp("t700_ns100000_ne50000_nr1000000_ts20_td1000", tsamp=20, M=45, xlim_acf_anal=100, xlim_acf=500) 
    
    # tsamp 50
    analyze_exp("t700_ns100000_ne50000_nr1000000_ts50_td1000", tsamp=50, M=70, xlim_acf_anal=300, xlim_acf=500) 
    
    # tsamp 100
    analyze_exp("t700_ns10000_ne5000_nr100000_ts100_td1000", tsamp=100, M=150, xlim_acf_anal=500, xlim_acf=500) 
    
    
if __name__ == "__main__":
    
    # plot_mean_forces("t300_ns100000_ne100000_nr1500000_ts10_td1000")
    store_exp([195])
    analyze_exp("t300_ns100000_ne100000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=500) 
    
    
    # analyze_exp("t300_ns100000_ne100000_nr1500000_ts50_td1000", tsamp=50, M=100, xlim_acf_anal=200, xlim_acf=300) 
    
    
    # analyze_exp("t300_ns100000_ne100000_nr1500000_ts10_td1000", tsamp=10, M=30, xlim_acf_anal=200, xlim_acf=500) # at L=140 messy
    
    # analysis_300K()
    # analysis_700K()