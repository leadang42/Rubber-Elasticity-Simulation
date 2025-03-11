import numpy as np
from statsmodels.tsa.stattools import acf
import os
import yaml
import shutil
import matplotlib.pyplot as plt
import emcee

# STORE SIMULATION DATA #
def store_config(L):
    
    input_filepath="lammps/lmp.input"
    
    # Extract variables
    target_vars = ['temp', 'Nstretch', 'Nequilib', 'Nrun', 'tsamp', 'tdump']
    values = {}
    
    with open(input_filepath, 'r') as file:
        for line in file:
            if line.startswith('variable'):
                parts = line.split()
                if parts[1] in target_vars and parts[2] == 'equal':
                    values[parts[1]] = int(float(parts[3])) # float to read 1eN notation and int to remove decimal point
    
    # Overwrite L with provided value
    values = {'L': L, **{k: v for k, v in values.items() if k != 'L'}}
    
    # Create experiment name and output directory
    exp_id = f"t{values['temp']}_ns{values['Nstretch']}_ne{values['Nequilib']}_nr{values['Nrun']}_ts{values['tsamp']}_td{values['tdump']}"
    sim_id = f"l{values['L']}_t{values['temp']}_ns{values['Nstretch']}_ne{values['Nequilib']}_nr{values['Nrun']}_ts{values['tsamp']}_td{values['tdump']}"

    output_dir = os.path.join('simulations', exp_id, sim_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to config to YAML
    output_path = os.path.join(output_dir, 'config.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(values, f, default_flow_style=False)
    
    return exp_id, values
   
def store_outputs(L, exp_id):
    
    sim_id = f"l{L}_{exp_id}"
    output_dir = os.path.join('simulations', exp_id, sim_id)
    os.makedirs(output_dir, exist_ok=True)
    
    for file in [f'lmp{L}.{ext}' for ext in ['xyz', 'forces', 'log']]:
        try:
            shutil.copy2(os.path.join('lammps', file), os.path.join(output_dir, file))
        except FileNotFoundError:
            print(f"Warning: {file} not found")
            
    return output_dir

def set_config(L=None, temp=None, Nstretch=None, Nequilib=None, Nrun=None, tsamp=None, tdump=None):
    filepath = "lammps/lmp.input"
    
    # Define parameter mapping
    param_map = {'L': L, 'temp': temp, 'Nstretch': Nstretch, 'Nequilib': Nequilib, 'Nrun': Nrun, 'tsamp': tsamp, 'tdump': tdump}
    
    # Read file contents
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Update each parameter if provided
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith('variable'):
            parts = line_stripped.split()
            if len(parts) >= 4 and parts[1] in param_map:
                param_name = parts[1]
                new_value = param_map[param_name]
                if new_value is not None:
                    lines[i] = f'variable        {param_name} equal {new_value}\n'
    
    # Write updated contents back to file
    with open(filepath, 'w') as f:
        f.writelines(lines)

# HELPER FUNCTIONS #
def autocorrelation(x):
    """Calculate autocorrelation function for a time series."""
    
    # Slice and normalize
    x_centered = x - np.mean(x)
    
    # Slice and normalize
    n = len(x)
    acf = np.correlate(x_centered, x_centered, mode='full') # Symmetric array of length 2n - 1
    acf = acf[n-1:] / (np.var(x) * n)  

    return acf

def jacobs_tau_int(ac, c=6.0, tol=0.01, max_iter=10):
    # Self-consistent estimate of τ: start with initial guess 1.0
    tau_est = 1.0
    for _ in range(max_iter):
        # Use a cutoff M = c * tau_est
        M = int(c * tau_est)
        # Sum the autocorrelation up to M (or until ac runs out)
        new_tau = 1.0 + 2.0 * np.sum(ac[1:min(len(ac), M)])
        if abs(new_tau - tau_est) / tau_est < tol:
            break
        tau_est = new_tau
    
    return tau_est

def integrated_corr_time(acf):
    
    # TODO adjust to step size
    
    if not np.isclose(acf[0], 1.0):
        raise ValueError("Autocorrelation function must be normalized to 1.0 at lag 0.")
    
    cutoff = 6.0        # Reasonable cutoff for autocorrelation sum
    tolerance = 0.005   # Strict stopping condition, high precision
    iterations = 15     # Slightly more iterations for stability

    tau_int = 1.0  # Initial estimate
    
    for _ in range(iterations):
        # Determine truncation limit
        M = min(len(acf), int(cutoff * tau_int))
        
        # Compute new estimate using truncated sum
        tau_int_new = 1.0 + 2.0 * np.sum(acf[1:M])
        
        # Check for convergence
        if abs(tau_int_new - tau_int) / tau_int < tolerance:
            return tau_int_new
        
        tau_int = tau_int_new 

    return tau_int  

# LECTURE CALCULATIONS #
def acf(force_data):
    N = len(force_data)
    mean_force = np.mean(force_data)
    max_k = N // 4  # Only use first quarter for reliable statistics
    
    # Compute unnormalized autocorrelation
    acf = np.zeros(max_k)
    for k in range(max_k):
        sum_corr = 0
        for i in range(N - k):
            sum_corr += (force_data[i] - mean_force) * (force_data[i + k] - mean_force)
        acf[k] = sum_corr / (N - k)
    
    # Normalize
    normalized_acf = acf / acf[0]
    
    return normalized_acf

def tau(acf, tsamp=10):
    
    # Calculate tau and convert to simulation steps
    tau = (1 + 2 * np.sum(acf[1:])) * tsamp
    
    return tau
    
# LECTURE CALCULATIONS #
def emcee_acf(x):
    return emcee.autocorr.function_1d(x)

def emcee_tau(x, tsamp):
    tau_samples = emcee.autocorr.integrated_time(x, c=5, tol=50, quiet=False, has_walkers=False)[0]  # One value in numpy array
    tau_steps = tau_samples * tsamp
    
    return tau_steps

# ANALYZE DATA # 
def plot_acf(acf, tau_int, L, exp_id):
    
    acf = acf[:1000]  # Limit to first 100 points
    
    t = np.arange(len(acf))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, acf, 'b-', label='Autocorrelation')
    
    # Add horizontal lines at y=0 and vertical line at tau_int
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=tau_int, color='r', linestyle='--', 
               label=f'τ_int = {tau_int:.2f}', alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Time lag (t)', fontsize=12)
    ax.set_ylabel('Autocorrelation C(t)', fontsize=12)
    ax.set_title('Autocorrelation Function', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(loc='upper right')
    
    # Set reasonable y-axis limits
    ax.set_ylim(-0.2, 1.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = f'simulations/{exp_id}/l{L}_{exp_id}/acf_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    return fig

def process_forces(L, exp_id, tsamp):
    
    sim_id = f"l{L}_{exp_id}"
    
    forces_filepath = f"simulations/{exp_id}/{sim_id}/lmp{L}.forces"
    forces = np.loadtxt(forces_filepath)
    
    # Calculate restoring forces
    fxh, fxt = forces[:, 0], forces[:, 3] 
    restoring_forces = fxh - fxt
    
    mean_force = np.mean(restoring_forces)
    var_force = np.var(restoring_forces)
    std_force = np.sqrt(var_force)
    
    # Analyze acf and tau
    #acf = autocorrelation(restoring_forces)
    #tau_int = jacobs_tau_int(acf)
    #tau_int = integrated_corr_time(acf) * 10 # For sampling
    
    acf = emcee_acf(restoring_forces)
    tau_int = emcee_tau(restoring_forces, tsamp=tsamp)
    
    plot_acf(acf, tau_int, L, exp_id)
    
    # Calculate effective number of independent samples
    N = len(restoring_forces)
    N_eff = N / (2 * tau_int)
    
    # Calculate error bar using error ~ std(A)/sqrt(N_eff)
    #error_bar = std_force / np.sqrt(N_eff)
    error_bar = std_force * np.sqrt((2 * tau_int) / N)
    
    # Create results dictionary
    results = {
        'displacement': L,
        'mean_force': float(mean_force),
        'std_force': float(np.sqrt(var_force)),
        'error_bar': float(error_bar),
        'tau_int': float(tau_int),
        'N_eff': float(N_eff),
    }
    
    # Save results to YAML
    forces_analysis_filepath = f"simulations/{exp_id}/{sim_id}/force_analysis.yaml"
    with open(forces_analysis_filepath, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    return results   

def plot_mean_forces(exp_id):
    
    # Initialize lists to store data
    displacements = []
    mean_forces = []
    error_bars = []
    
    # Search through all simulation directories
    sim_base_dir = f'simulations/{exp_id}'
    
    for dirname in os.listdir(sim_base_dir):
        
        # Check if directory matches our parameter set (Skip the "l" prefix and check if rest matches)
        if dirname.startswith('l') and dirname[dirname.find('_')+1:] == exp_id:
            exp_dir = os.path.join(sim_base_dir, dirname)
            
            # Read the force analysis results
            yaml_path = os.path.join(exp_dir, 'force_analysis.yaml')
            
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    results = yaml.safe_load(f)
                    
                    displacements.append(results['displacement'])
                    mean_forces.append(results['mean_force'])
                    error_bars.append(results['error_bar'])
    
    if not displacements:
        raise ValueError(f"No data found for parameter set: {exp_id}")
    
    # Convert to numpy arrays for easier manipulation
    displacements = np.array(displacements)
    mean_forces = np.array(mean_forces)
    error_bars = np.array(error_bars)
    
    # Sort data by displacement
    sort_idx = np.argsort(displacements)
    displacements = displacements[sort_idx]
    mean_forces = mean_forces[sort_idx]
    error_bars = error_bars[sort_idx]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points with error bars
    ax.errorbar(displacements, mean_forces, yerr=error_bars, 
                fmt='o-', capsize=5, capthick=1.5, 
                elinewidth=1.5, markersize=8,
                label='Mean Force')
    
    # Customize the plot
    ax.set_xlabel('Displacement (L)', fontsize=12)
    ax.set_ylabel('Mean Force', fontsize=12)
    ax.set_title(f'Force vs Displacement\n{exp_id}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    
    plt.savefig(f"simulations/{exp_id}/force_plot.png", dpi=300, bbox_inches='tight')
    
    return fig

# AGGREGATE FUNCTION #
def process_experiment(Ls, save_results=True, save_forces=True, exp_id=None, tsamp=10):
    print(f"Processing experiment for {Ls}")
    
    for L in Ls:
        
        print(f"\nDisplacement L={L}:")
        
        if save_results:
            exp_id, values = store_config(L)
            tsamp = values["tsamp"]
            store_outputs(L, exp_id)
            
            print(f"Stored configuration and outputs")
        
        if save_forces:
            results = process_forces(L, exp_id, tsamp)
            
            print(f"Mean force: {results['mean_force']:.6f} ± {results['error_bar']:.6f}")
            print(f"Integrated correlation time: {results['tau_int']:.2f}")
    
    plot_mean_forces(exp_id)  
    return exp_id

if __name__ == "__main__":
    
    #process_experiment(Ls=[L for L in range(150, 280, 5)], save_results=False, save_forces=True, exp_id="t700_ns10000_ne5000_nr1000000_ts10_td1000",)
    process_experiment(Ls=[L for L in range(150, 290, 10)])
    #plot_mean_forces("t700_ns100000_ne50000_nr1000000_ts20_td1000")