"""Langevin dynamics of polyisoprene fragments with the MACE-OFF machine-learned force field.

    python exercises/exercises.py 1    # repeat unit at 100, 300, 500 and 700 K
    python exercises/exercises.py 3    # stretching a hexamer

Each run is defined by a YAML file in configs/ and writes its trajectory, distance
record and plots to <base_directory>/<experiment_name>/ (simulations/ by default).
"""

import argparse
from pathlib import Path

from ase.data.pubchem import pubchem_atoms_search
from mace.calculators import mace_off

from analysis import analyze_distances, analyze_stretches
from simulation import MolecularDynamics, SimulationConfig

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Hydrogen-terminated polyisoprene repeat unit, and a hexamer of it
MONOMER_SMILES = "CC=C(C)C"
HEXAMER_SMILES = "CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C"


def load_molecule(smiles):
    """Fetch a structure from PubChem and attach the MACE-OFF calculator."""
    molecule = pubchem_atoms_search(smiles=smiles)
    molecule.calc = mace_off(model="medium", device='cpu')
    return molecule


def run_simulation(smiles, config_name):
    """Run configs/<config_name>.yaml on a freshly loaded molecule and plot the results."""
    config = SimulationConfig.from_yaml(CONFIG_DIR / f"{config_name}.yaml")
    simulation = MolecularDynamics(load_molecule(smiles), config)
    simulation.run()

    analyze_distances(simulation.output_dir)
    analyze_stretches(simulation.output_dir)


def exercise_1(temperatures=(100, 300, 500, 700)):
    """Dynamics of a single repeat unit at different temperatures."""
    for temp in temperatures:
        run_simulation(MONOMER_SMILES, f"ex1_dynamics_10000s_{temp}K")


def exercise_3():
    """Stretching a hexamer, with a fixed stretch step and with a cumulative one."""
    run_simulation(HEXAMER_SMILES, "ex3_stretch_linear_1000s_pos1fixed")
    run_simulation(HEXAMER_SMILES, "ex3_stretch_cumulative_1000s_pos1pos2fixed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the ASE + MACE-OFF exercises.")
    parser.add_argument("exercise", type=int, choices=[1, 3], help="exercise to run")
    args = parser.parse_args()

    {1: exercise_1, 3: exercise_3}[args.exercise]()
