import numpy as np
import torch
import ase
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.constraints import FixAtoms
import pandas as pd
import os
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any

torch.set_default_dtype(torch.float64)

@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters"""
    experiment_name: str
    output_base_dir: str
    random_seed: Optional[int]
    timestep: float
    steps: int
    write_interval: int
    temperature: float
    friction: float
    atoms: Dict[str, Dict[str, Any]]
    stretching: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str) -> 'SimulationConfig':
        """Create config from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            experiment_name=config['experiment_name'],
            output_base_dir=config['base_directory'],
            random_seed=config['random_seed'],
            timestep=config['time']['timestep'],
            steps=config['time']['steps'],
            write_interval=config['time']['write_interval'],
            temperature=config['dynamics']['temp'],
            friction=config['dynamics']['friction'],
            atoms=config['atoms'],
            stretching=config['stretching']
        )

class MolecularDynamics:
    """Class to handle molecular dynamics simulation"""
    
    def __init__(self, molecule, config: SimulationConfig):
        # Simulation initialization variables
        self.molecule = molecule
        self.config = config
        
        # Simulation setup
        self.setup_output_files()
        self.setup_simulation()
        
        # Initialize tracking variables
        self.initial_distance = self._calculate_distance()
        self.cumulative_stretch = 0.0
        
    def setup_output_files(self):
        """Set up output directory and initialize output files"""
        # Create output directory
        self.output_dir = os.path.join(self.config.output_base_dir, self.config.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define all output paths
        self.xyz_path = os.path.join(self.output_dir, "positions.xyz")
        self.csv_distances_path = os.path.join(self.output_dir, "distances.csv")
        self.csv_stretches_path = os.path.join(self.output_dir, "stretches.csv")
        
        # Initialize trajectory file
        open(self.xyz_path, 'w').close()
        
        # Initialize distances CSV
        pd.DataFrame(columns=['time', 'distance']).to_csv(self.csv_distances_path, index=False)
        
        # Initialize stretches CSV if stretching is enabled
        if self.config.stretching['enable']:
            pd.DataFrame(columns=['time', 'cumulative_stretch']).to_csv(
                self.csv_stretches_path, index=False
            )
    
    def setup_simulation(self):
        """Initialize simulation parameters"""
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Set atom constraints
        constraints = []
        for pos in ['position1', 'position2']:
            if self.config.atoms[pos]['fixed']:
                constraints.append(FixAtoms(indices=[self.config.atoms[pos]['index']]))
        
        if constraints:
            self.molecule.set_constraint(constraints)
        
        # Initialize dynamics
        MaxwellBoltzmannDistribution(
            self.molecule, 
            temperature_K=self.config.temperature
        )
        
        self.dynamics = Langevin(
            self.molecule,
            temperature_K=self.config.temperature,
            timestep=self.config.timestep * units.fs,
            friction=self.config.friction
        )
    
    def _calculate_distance(self) -> float:
        """Calculate distance between specified atoms"""
        pos1 = self.config.atoms['position1']['index']
        pos2 = self.config.atoms['position2']['index']
        return np.linalg.norm(self.molecule.get_positions()[pos1] - self.molecule.get_positions()[pos2])
    
    def _record_data(self):
        """Record current simulation state"""
        current_time = float(self.dynamics.get_time() / units.fs)
        current_distance = self._calculate_distance()
        
        # Record to CSV
        pd.DataFrame({
            'time': [current_time],
            'distance': [current_distance]
        }).to_csv(self.csv_distances_path, mode='a', header=False, index=False)
        
        print(f"Time: {current_time:.1f} fs | Distance: {current_distance:.3f} Å")
        
        # Write trajectory
        with open(self.xyz_path, 'a') as f:
            ase.io.write(f, self.molecule, format="extxyz")
    
    def _stretch_molecule(self):
        """Apply stretching with optional cumulative mode and constraint handling."""
        if not self.config.stretching['enable']:
            return

        pos1, pos2 = self.config.atoms['position1']['index'], self.config.atoms['position2']['index']
        positions = self.molecule.get_positions()
        direction = positions[pos2] - positions[pos1]
        current_distance = np.linalg.norm(direction)

        # Determine stretch distance
        stretch_distance = (current_distance if self.config.stretching['cumulative'] else self.initial_distance) * self.config.stretching['amount']
        self.cumulative_stretch += stretch_distance

        # Temporarily unfix pos2 if necessary
        if self.config.atoms['position2']['fixed']:
            self.molecule.set_constraint([FixAtoms(indices=[pos1])])

        # Apply stretch
        positions[pos2] = positions[pos1] + (direction / current_distance) * (current_distance + stretch_distance)
        self.molecule.set_positions(positions)

        # Re-fix pos2 if necessary
        if self.config.atoms['position2']['fixed']:
            self.molecule.set_constraint(FixAtoms(indices=[pos1, pos2]))

        # Record data
        pd.DataFrame({
            'time': [self.dynamics.get_time() / units.fs], 
            'cumulative_stretch': [self.cumulative_stretch]
        }).to_csv(self.csv_stretches_path, mode='a', header=False, index=False)

        print(f"Time: {self.dynamics.get_time() / units.fs:.1f} fs | Stretch applied: {self.cumulative_stretch:.3f} Å")

    
    def run(self):
        """Run the simulation"""
        self.dynamics.attach(self._record_data, interval=self.config.write_interval)
        self.dynamics.attach(self._stretch_molecule, interval=self.config.stretching['interval'])
        self.dynamics.run(self.config.steps)
        self.dynamics.close()
