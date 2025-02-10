from ase.data.pubchem import pubchem_atoms_search
from mace.calculators import mace_off
from src.simulation import SimulationConfig, MolecularDynamics
from src.analysis import analyze_distances, analyze_stretches

def exercise_3():
    """
    Loads polyisoprene molecule, uses the mace energy model and calculates
    """
    polyisoprene = pubchem_atoms_search(smiles="CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C")
    polyisoprene.calc = mace_off(model="medium", device='cpu') 

    config = SimulationConfig.from_yaml("simulations/ex3_1000_steps_cumulative_stretching_pos2fixed/config.yaml")
    simulation = MolecularDynamics(polyisoprene, config)
    simulation.run()

if __name__ == '__main__':
    #simulation_path = "simulations/ex3_1000_steps_stretching"
    #analyze_distances(simulation_path)
    #analyze_stretches(simulation_path)
    
    exercise_3()