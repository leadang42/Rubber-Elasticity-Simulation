"""Post-processing of the LAMMPS polyisoprene stretching simulations.

Workflow (see the README for details):

    python analysis/lammps_analysis.py store              # collect lammps/lmp<L>.* into simulations/
    python analysis/lammps_analysis.py analyze EXP_ID -M 30
    python analysis/lammps_analysis.py plot EXP_ID        # redraw force/energy vs. L plots

Each set of runs sharing the same parameters in lammps/lmp.input is an "experiment",
identified by an ID such as ``t300_ns100000_ne100000_nr1500000_ts10_td1000``.
"""

import argparse
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.cm import viridis
from statsmodels.tsa.stattools import acf

REPO_ROOT = Path(__file__).resolve().parents[1]
LAMMPS_DIR = REPO_ROOT / "lammps"
SIMULATIONS_DIR = REPO_ROOT / "simulations"

# Run parameters that define an experiment: (ID prefix, lmp.input variable)
ID_FIELDS = [
    ("t", "temp"),
    ("ns", "Nstretch"),
    ("ne", "Nequilib"),
    ("nr", "Nrun"),
    ("ts", "tsamp"),
    ("td", "tdump"),
]
ID_PATTERN = re.compile("_".join(rf"{prefix}(\d+)" for prefix, _ in ID_FIELDS))

FORCE_COLOR, FORCE_ERROR_COLOR = viridis(0.1), viridis(0.3)
ENERGY_COLOR, ENERGY_ERROR_COLOR = viridis(0.7), viridis(0.8)


# ===== EXPERIMENT BOOKKEEPING =====

def parse_input_file(input_file=LAMMPS_DIR / "lmp.input"):
    """Extract the run parameters from the LAMMPS input file."""
    names = {name for _, name in ID_FIELDS}
    params = {}
    with open(input_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "variable" and parts[1] in names and parts[2] == "equal":
                params[parts[1]] = int(float(parts[3]))
    return params


def experiment_id(params):
    """Build the experiment ID from the run parameters."""
    return "_".join(f"{prefix}{params[name]}" for prefix, name in ID_FIELDS)


def parse_experiment_id(exp_id):
    """Recover the run parameters from an experiment ID."""
    match = ID_PATTERN.fullmatch(exp_id)
    if match is None:
        raise ValueError(f"Invalid experiment ID '{exp_id}', expected e.g. t300_ns100000_ne100000_nr1500000_ts10_td1000")
    return {name: int(value) for (_, name), value in zip(ID_FIELDS, match.groups())}


def simulation_dir(exp_id, L):
    """Directory holding the outputs of one simulation (chain length L) of an experiment."""
    return SIMULATIONS_DIR / exp_id / f"l{L}_{exp_id}"


def find_simulations(exp_id):
    """Find all simulations of an experiment, as (L, directory) pairs sorted by L."""
    exp_dir = SIMULATIONS_DIR / exp_id
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    simulations = []
    for path in exp_dir.iterdir():
        prefix, _, rest = path.name.partition("_")
        if path.is_dir() and prefix.startswith("l") and rest == exp_id:
            simulations.append((int(prefix[1:]), path))
    return sorted(simulations, key=lambda s: s[0])


def find_lammps_outputs():
    """Chain lengths L for which LAMMPS output (lmp<L>.forces) exists in lammps/."""
    lengths = []
    for path in LAMMPS_DIR.glob("lmp*.forces"):
        match = re.fullmatch(r"lmp(\d+)\.forces", path.name)
        if match:
            lengths.append(int(match.group(1)))
    return sorted(lengths)


def store_experiment(lengths):
    """Copy the LAMMPS output for each chain length L into the experiment directory."""
    params = parse_input_file()
    exp_id = experiment_id(params)

    for L in lengths:
        sim_dir = simulation_dir(exp_id, L)
        sim_dir.mkdir(parents=True, exist_ok=True)

        with open(sim_dir / "config.yaml", "w") as f:
            yaml.dump({**params, "L": L}, f, default_flow_style=False)

        for suffix in ("xyz", "forces", "log"):
            src = LAMMPS_DIR / f"lmp{L}.{suffix}"
            if src.exists():
                shutil.copy2(src, sim_dir / src.name)
            else:
                print(f"Warning: {src} not found")

        print(f"-> Stored configuration and outputs for L={L}")

    return exp_id


# ===== DATA LOADING =====

def load_forces(path):
    """Load the end-atom forces written during the production run.

    Columns: fxh fyh fzh fxt fyt fzt xh xt (h = head atom, t = tail atom).
    """
    return np.loadtxt(path)


def load_thermo(path):
    """Load the thermo output (Step, Temp, PotEng) of all runs in a LAMMPS log."""
    data = []
    in_thermo_section = False

    with open(path) as f:
        for line in f:
            line = line.strip()

            if "Step" in line and "Temp" in line and "PotEng" in line:
                in_thermo_section = True
                continue

            if in_thermo_section:
                if line.startswith("Loop time") or not line:
                    in_thermo_section = False
                    continue

                values = line.split()
                if len(values) >= 3:
                    data.append([float(v) for v in values[:3]])

    return np.array(data) if data else np.empty((0, 3))


# ===== STATISTICS =====

def restoring_force(forces):
    """Restoring force along the stretching (x) axis."""
    return forces[:, 0] - forces[:, 3]


def autocorrelation(x):
    """Normalised autocorrelation function of a time series for all lags."""
    return acf(x, nlags=len(x) - 1, fft=True, adjusted=False)


def integrated_autocorrelation_time(rho, tsamp, M):
    """Integrated autocorrelation time in MD steps, summing the ACF up to lag M."""
    M = min(M, len(rho) - 1)
    return (0.5 + np.sum(rho[1:M + 1])) * tsamp


def analyze_series(series, quantity, L, tsamp, M):
    """Mean, standard deviation and correlation-corrected error bar of a time series."""
    rho = autocorrelation(series)
    tau_int = integrated_autocorrelation_time(rho, tsamp, M)
    std = np.std(series)
    N = len(series)

    results = {
        "displacement": L,
        f"mean_{quantity}": float(np.mean(series)),
        f"std_{quantity}": float(std),
        "error_bar": float(np.sqrt(2 * tau_int / N) * std),
        "tau_int": float(tau_int),
        "N_eff": float(N / (2 * tau_int)),
        "mean_acf": float(np.mean(rho)),
        "M": M,
    }
    return results, rho


# ===== PLOTTING =====

def format_steps(n):
    """Compact notation for step counts, e.g. 100000 -> 1e5, 1500000 -> 1.5e6."""
    if n < 1000:
        return str(n)
    mantissa, exponent = f"{n:e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}"


def format_experiment_title(exp_id):
    """Format an experiment ID into a readable plot title."""
    try:
        p = parse_experiment_id(exp_id)
    except ValueError:
        return exp_id
    return (f"{p['temp']}K - Stretch {format_steps(p['Nstretch'])} - Equil {format_steps(p['Nequilib'])}"
            f" - Prod {format_steps(p['Nrun'])} - Tsamp {p['tsamp']}")


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(series, tsamp, ylabel, color, output_path):
    """Plot a raw time series against simulation time."""
    time_steps = np.arange(len(series)) * tsamp

    fig, ax = plt.subplots(figsize=(7, 2))
    ax.plot(time_steps, series, color=color, linewidth=1)
    ax.set_xlabel("Simulation Time", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim([0, time_steps[-1]])
    ax.grid(True, linestyle=":", alpha=0.4)
    save_figure(fig, output_path)


def plot_autocorrelation(rho, xlim, output_path):
    """Plot the autocorrelation function."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(rho)), rho, color=viridis(0.2), label="Autocorrelation")
    ax.set_xlabel(r"Time lag $\tau$", fontsize=12)
    ax.set_ylabel(r"Autocorrelation $\rho(\tau)$", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim([0, xlim])
    ax.set_ylim([-0.5, 0.5])
    ax.legend(loc="best")
    save_figure(fig, output_path)


def plot_autocorrelation_halves(x, xlim, output_path):
    """Compare the autocorrelation functions of the first and second half of a time series."""
    half = len(x) // 2
    rho_first = autocorrelation(x[:half])
    rho_second = autocorrelation(x[half:])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(len(rho_first)), rho_first, color=viridis(0.2), label="First Half")
    ax.plot(np.arange(len(rho_second)), rho_second, color=viridis(0.6), label="Second Half", alpha=0.8)
    ax.set_xlabel(r"Time lag $\tau$", fontsize=12)
    ax.set_ylabel(r"Autocorrelation $\rho(\tau)$", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_ylim([-0.5, 1])
    ax.set_xlim([0, xlim])
    ax.legend(loc="best")
    save_figure(fig, output_path)


def load_experiment_results(exp_id, quantity):
    """Collect (L, mean, error bar) of 'force' or 'energy' for all simulations of an experiment."""
    rows = []
    for _, sim_dir in find_simulations(exp_id):
        path = sim_dir / f"{quantity}_analysis.yaml"
        if path.exists():
            with open(path) as f:
                results = yaml.safe_load(f)
            rows.append((results["displacement"], results[f"mean_{quantity}"], results["error_bar"]))

    if not rows:
        raise FileNotFoundError(f"No {quantity}_analysis.yaml found for experiment {exp_id}, run 'analyze' first")
    return sorted(rows)


def errorbar(ax, rows, fmt, color, ecolor, label):
    displacements, means, error_bars = zip(*rows)
    ax.errorbar(
        displacements, means, yerr=error_bars,
        fmt=fmt, capsize=5, capthick=1.5, elinewidth=1.5, markersize=5,
        color=color, ecolor=ecolor, mfc=color, mec=ecolor, label=label,
    )


def plot_experiment(exp_id, kind):
    """Plot mean force ('force'), energy ('energy') or both ('combined') against L."""
    print(f"-> Generating {kind} plot for experiment: {exp_id}")
    title = format_experiment_title(exp_id)
    fig, ax = plt.subplots(figsize=(7, 7))

    if kind == "combined":
        errorbar(ax, load_experiment_results(exp_id, "force"), "o-",
                 FORCE_COLOR, FORCE_ERROR_COLOR, "Mean Restoring Force")
        ax.set_xlabel("Displacement [Å]", fontsize=12)
        ax.set_ylabel("Mean Restoring Force [eV/Å]", fontsize=12, color=FORCE_COLOR)
        ax.tick_params(axis="y", labelcolor=FORCE_COLOR)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Energy on the right y-axis
        ax2 = ax.twinx()
        errorbar(ax2, load_experiment_results(exp_id, "energy"), "s--",
                 ENERGY_COLOR, ENERGY_ERROR_COLOR, "Mean Internal Energy")
        ax2.set_ylabel("Mean Internal Energy [eV]", fontsize=12, color=ENERGY_COLOR)
        ax2.tick_params(axis="y", labelcolor=ENERGY_COLOR)

        ax.set_title(f"Force and Internal Energy vs Displacement\n{title}", fontsize=14)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    elif kind == "force":
        errorbar(ax, load_experiment_results(exp_id, "force"), "o-",
                 FORCE_COLOR, FORCE_ERROR_COLOR, "Mean Restoring Force")
        ax.set_ylabel("Mean Restoring Force [eV/Å]", fontsize=12)
        ax.set_title(f"Force vs Displacement\n{title}", fontsize=14)

    elif kind == "energy":
        errorbar(ax, load_experiment_results(exp_id, "energy"), "o-",
                 ENERGY_COLOR, ENERGY_ERROR_COLOR, "Mean Internal Energy")
        ax.set_ylabel("Mean Internal Energy [eV]", fontsize=12)
        ax.set_title(f"Internal Energy vs Displacement\n{title}", fontsize=14)

    else:
        raise ValueError(f"Unknown plot kind: {kind}")

    if kind != "combined":
        ax.set_xlabel("Displacement [Å]", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

    save_figure(fig, SIMULATIONS_DIR / exp_id / f"{kind}_plot.png")


def plot_experiment_summary(exp_id):
    for kind in ("force", "energy", "combined"):
        plot_experiment(exp_id, kind)


# ===== ANALYSIS =====

def analyze_experiment(exp_id, M, xlim_acf=500, xlim_acf_halves=200):
    """Compute force and energy statistics for every chain length L of an experiment.

    M is the ACF summation window (in samples) for the integrated autocorrelation time;
    xlim_acf and xlim_acf_halves only set the lag range shown in the ACF plots.
    """
    params = parse_experiment_id(exp_id)
    tsamp = params["tsamp"]
    # Thermo output is sampled every tsamp steps from the start of the stretching stage
    run_start = int((params["Nstretch"] + params["Nequilib"]) / tsamp)

    for L, sim_dir in find_simulations(exp_id):
        forces = restoring_force(load_forces(sim_dir / f"lmp{L}.forces"))
        energy = load_thermo(sim_dir / f"lmp{L}.log")[run_start:, 2]

        force_results, force_acf = analyze_series(forces, "force", L, tsamp, M)
        energy_results, _ = analyze_series(energy, "energy", L, tsamp, M)

        for name, results in (("force", force_results), ("energy", energy_results)):
            with open(sim_dir / f"{name}_analysis.yaml", "w") as f:
                yaml.dump(results, f, default_flow_style=False)

            print(f"-> {name.capitalize()} Analysis for L={L}:")
            for key, value in results.items():
                if key != "displacement":
                    print(f"   - {key}: {value}")
        print()

        plot_autocorrelation(force_acf, xlim_acf, sim_dir / "acf_plot.png")
        plot_autocorrelation_halves(forces, xlim_acf_halves, sim_dir / "acf_plot_halfs.png")
        plot_time_series(forces, tsamp, "Restoring Force", viridis(0.3), sim_dir / "force_time_plot.png")
        plot_time_series(energy, tsamp, "Internal Energy", viridis(0.6), sim_dir / "energy_time_plot.png")

    plot_experiment_summary(exp_id)


# ===== COMMAND LINE =====

def main(argv=None):
    parser = argparse.ArgumentParser(description="Post-process the LAMMPS polyisoprene stretching simulations.")
    commands = parser.add_subparsers(dest="command", required=True)

    store = commands.add_parser("store", help="copy LAMMPS output from lammps/ into simulations/<experiment ID>/")
    store.add_argument("lengths", nargs="*", type=int, metavar="L",
                       help="chain lengths to store (default: every lmp<L>.forces in lammps/)")

    analyze = commands.add_parser("analyze", help="compute mean force/energy with error bars and plot them against L")
    analyze.add_argument("exp_id", help="experiment ID, e.g. t300_ns100000_ne100000_nr1500000_ts10_td1000")
    analyze.add_argument("-M", type=int, required=True,
                         help="ACF summation window (in samples) for the integrated autocorrelation time")
    analyze.add_argument("--xlim-acf", type=int, default=500, help="lag range shown in acf_plot.png (default: 500)")
    analyze.add_argument("--xlim-acf-halves", type=int, default=200,
                         help="lag range shown in acf_plot_halfs.png (default: 200)")

    plot = commands.add_parser("plot", help="regenerate the force/energy vs. L plots of an analysed experiment")
    plot.add_argument("exp_id", help="experiment ID")

    args = parser.parse_args(argv)

    if args.command == "store":
        lengths = args.lengths or find_lammps_outputs()
        if not lengths:
            parser.error(f"no lmp<L>.forces files found in {LAMMPS_DIR}, run ./run_lammps.sh first")
        exp_id = store_experiment(lengths)
        print(f"\nExperiment ID: {exp_id}")
    elif args.command == "analyze":
        analyze_experiment(args.exp_id, args.M, args.xlim_acf, args.xlim_acf_halves)
    elif args.command == "plot":
        plot_experiment_summary(args.exp_id)


if __name__ == "__main__":
    main()
