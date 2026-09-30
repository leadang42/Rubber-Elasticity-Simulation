# Rubber Elasticity Simulation

Molecular dynamics simulations of stretched polyisoprene, the polymer in natural rubber. A single
50-unit *cis*-1,4-polyisoprene chain is held at a series of end-to-end lengths *L* in
[LAMMPS](https://www.lammps.org), and the mean restoring force and potential energy are measured
at each length. Together they show how much of the chain's elasticity is energetic and how much
is entropic. Warm-up exercises with the [MACE-OFF](https://github.com/ACEsuit/mace)
machine-learned force field in [ASE](https://wiki.fysik.dtu.dk/ase/) explore the dynamics of
short polyisoprene fragments.

## Background

Rubber owes its elasticity mostly to entropy. A stretched chain can adopt fewer conformations,
so its free energy rises even when its internal energy barely changes. At constant temperature
the tension of a chain held at end-to-end length $L$ is

$$
f = \left(\frac{\partial A}{\partial L}\right)_T
  = \left(\frac{\partial U}{\partial L}\right)_T - T\left(\frac{\partial S}{\partial L}\right)_T .
$$

Measuring both $\langle f \rangle$ and $\langle U \rangle$ as functions of $L$ separates the
energetic contribution $\partial U/\partial L$ from the entropic one $-T\,\partial S/\partial L$.

## Repository structure

```
├── lammps/                  LAMMPS input deck
│   ├── lmp.input            simulation protocol and run parameters
│   ├── lmp.param            force-field parameters
│   └── lmp.data             initial structure of the 50-unit chain
├── run_lammps.sh            runs LAMMPS (via Docker) for a series of lengths L
├── analysis/
│   ├── lammps_analysis.py   collects runs, computes forces/energies with error bars, plots vs. L
│   └── experiments.py       analysis settings used for each simulation campaign
├── exercises/               ASE + MACE-OFF warm-up exercises
│   ├── exercises.py         entry point
│   ├── simulation.py        Langevin MD driver with optional stretching
│   ├── analysis.py          distance, stretch and force analysis of trajectories
│   └── configs/             one YAML file per simulation
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/leadang42/Rubber-Elasticity-Simulation.git
cd Rubber-Elasticity-Simulation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The LAMMPS simulations run in the official [`lammps/lammps`](https://hub.docker.com/r/lammps/lammps)
Docker image (LAMMPS 30 Jul 2021), so apart from Python you only need
[Docker](https://docs.docker.com/get-docker/). The analysis needs only numpy, matplotlib, pyyaml
and statsmodels. `mace-torch`, which pulls in PyTorch, is needed only for the exercises.

Run the commands below from the repository root.

## LAMMPS force–extension simulations

### Model

- **Chain:** 50 repeat units of *cis*-1,4-polyisoprene capped with hydrogen (652 atoms), alone in
  a 750 Å non-periodic box. The structure in `lmp.data` was written with ASE's OPLS tools.
- **Force field** (`lmp.param`): OPLS-style, with harmonic bonds and angles, OPLS dihedrals, and
  Lennard-Jones + Coulomb interactions cut off at 12 Å (1–4 pairs scaled by 0.5). There are three
  atom types: sp³ carbon, sp² carbon and hydrogen.
- **Dynamics:** LAMMPS `metal` units (Å, eV, ps), 1 fs time step, Nosé–Hoover NVT thermostat.

### Protocol

For each end-to-end length *L*, `lmp.input` runs three stages. The chain ends are the sp² carbons
of the first and last repeat units (atom IDs 2 and 642).

1. **Stretch** (`Nstretch` steps): the two end atoms move at constant velocity until they are
   separated by exactly *L* along *x*.
2. **Equilibrate** (`Nequilib` steps): the end atoms are held fixed.
3. **Production** (`Nrun` steps): with the ends still fixed, the forces on the end atoms are
   written to `lmp<L>.forces` every `tsamp` steps.

Throughout all three stages, the temperature and potential energy are logged to `lmp<L>.log`
every `tsamp` steps, and coordinates are written to `lmp<L>.xyz` every `tdump` steps.

The run parameters are set at the top of `lammps/lmp.input`:

| Variable   | Meaning                                    | Value in `lmp.input` |
| ---------- | ------------------------------------------ | -------------------- |
| `temp`     | temperature (K)                            | `700`                |
| `Nstretch` | steps to bring the ends to separation *L*  | `1e5`                |
| `Nequilib` | equilibration steps at fixed *L*           | `1e5`                |
| `Nrun`     | production steps                           | `1500000`            |
| `tsamp`    | steps between force/energy samples         | `10`                 |
| `tdump`    | steps between trajectory frames            | `1000`               |

### Running

```bash
./run_lammps.sh              # L = 140, 145, ..., 200 Å
./run_lammps.sh 150 175      # selected lengths only
```

Each run writes `lmp<L>.log`, `lmp<L>.forces` and `lmp<L>.xyz` to `lammps/`. The Docker image
is amd64-only, so on Apple silicon it runs under emulation. There, a full run of 1.7 × 10⁶ steps
takes about 15 minutes per length on 4 threads. The image and thread count can be changed with
the `LAMMPS_IMAGE` and `OMP_NUM_THREADS` environment variables. With a local LAMMPS
installation, the equivalent of one run is `cd lammps && lmp -in lmp.input -var Lval 150`.

### Analysis

All runs made with one set of parameters form an *experiment*. Its ID is built from the values
in `lmp.input`, for example `t300_ns100000_ne100000_nr1500000_ts10_td1000`:
`t<temp>_ns<Nstretch>_ne<Nequilib>_nr<Nrun>_ts<tsamp>_td<tdump>`.

```bash
# 1. Copy lammps/lmp<L>.* into simulations/<experiment ID>/ and print the ID
python analysis/lammps_analysis.py store

# 2. Compute force and energy statistics for every L, then plot them against L
python analysis/lammps_analysis.py analyze t300_ns100000_ne100000_nr1500000_ts10_td1000 -M 30

# 3. (Optional) Redraw the summary plots
python analysis/lammps_analysis.py plot t300_ns100000_ne100000_nr1500000_ts10_td1000
```

`store` names the experiment from the parameters currently in `lmp.input`, so run it before
editing `lmp.input` for the next campaign.

For every *L*, `analyze` takes the restoring force $f = f_x^\mathrm{head} - f_x^\mathrm{tail}$
from the production forces and the potential energy from the production part of the log. The
plots call the potential energy "internal energy": at fixed temperature the mean kinetic energy
is constant, so changes in *U* come from the potential energy alone.

Consecutive samples are correlated, so the error bar of each mean is $\sigma\sqrt{2\tau/N}$.
Here $\sigma$ is the standard deviation of the $N$ samples, and
$\tau = \tfrac{1}{2} + \sum_{k=1}^{M} \rho(k)$ is the integrated autocorrelation time in units
of the sampling interval, where $\rho$ is the normalised autocorrelation function.
$N/(2\tau)$ is the effective number of independent samples. Choose `M` where `acf_plot.png`
has decayed into noise. The values used for each campaign are recorded in
`analysis/experiments.py`. The output files report `tau_int` in MD steps, that is
$\tau \times$ `tsamp`.

Results are written to `simulations/`, which is not tracked by git:

```
simulations/<experiment ID>/
├── force_plot.png, energy_plot.png, combined_plot.png     mean force / energy vs. L
└── l<L>_<experiment ID>/
    ├── config.yaml                                        run parameters
    ├── lmp<L>.log, lmp<L>.forces, lmp<L>.xyz              raw LAMMPS output
    ├── force_analysis.yaml, energy_analysis.yaml          mean, std, error bar, τ_int, N_eff
    └── acf_plot.png, acf_plot_halfs.png,                  autocorrelation diagnostics and
        force_time_plot.png, energy_time_plot.png          raw time series
```

## ASE + MACE-OFF exercises

These are short Langevin dynamics simulations of polyisoprene fragments with the MACE-OFF23
(medium) force field in ASE.

```bash
python exercises/exercises.py 1    # repeat unit (SMILES CC=C(C)C) at 100, 300, 500 and 700 K
python exercises/exercises.py 3    # hexamer stretched by 2 % every 10 steps
```

Exercise 2, setting up the hexamer with MACE-OFF, is part of exercise 3. The structures are
downloaded from PubChem and the MACE-OFF model is downloaded on first use, so the exercises
need internet access.

Each simulation is described by a YAML file in `exercises/configs/`:

| Key                          | Meaning                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `experiment_name`            | name of the output directory `<base_directory>/<experiment_name>/`           |
| `base_directory`             | parent directory for the output, relative to the working directory           |
| `random_seed`                | seed for the initial velocities and the thermostat                           |
| `dynamics.temp`              | temperature (K)                                                              |
| `dynamics.friction`          | Langevin friction coefficient (ASE units)                                    |
| `time.timestep`              | time step (fs)                                                               |
| `time.steps`                 | number of MD steps                                                           |
| `time.write_interval`        | steps between recorded frames and distances                                  |
| `atoms.position1/position2`  | `index` of each atom in the tracked pair, and whether it is `fixed`          |
| `stretching.enable`          | pull the pair apart during the run                                           |
| `stretching.interval`        | steps between stretches                                                      |
| `stretching.amount`          | each stretch moves `position2` outwards by `amount` × the pair distance      |
| `stretching.cumulative`      | use the current distance (`true`) or the initial distance (`false`)          |

Each output directory contains `positions.xyz` (an extended XYZ trajectory including forces),
`distances.csv`, `stretches.csv` (only for stretching runs) and plots in `plots/`.
