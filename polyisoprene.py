from ase.data.pubchem import pubchem_atoms_search
from mace.calculators import mace_off
from src.simulation import SimulationConfig, MolecularDynamics
from src.analysis import analyze_distances, analyze_stretches, get_direction_vector

def exercise_1():
    isoprene = pubchem_atoms_search(smiles="CC=C(C)C")
    isoprene.calc = mace_off(model="medium", device='cpu') 
    
    temps = [500, 700]
    
    for temp in temps: 
        config_path = f"simulations/ex1_dynamics_10000s_{temp}K/config.yaml"
        simulation_path = f"simulations/ex1_dynamics_10000s_{temp}K"
        
        simulation = MolecularDynamics(isoprene, SimulationConfig.from_yaml(config_path))
        simulation.run()
        
        analyze_distances(simulation_path)
        analyze_stretches(simulation_path)
    
def exercise_2():
    polyisoprene = pubchem_atoms_search(smiles="CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C")
    polyisoprene.calc = mace_off(model="medium", device='cpu') 

def exercise_3():
    polyisoprene = pubchem_atoms_search(smiles="CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C")
    polyisoprene.calc = mace_off(model="medium", device='cpu') 

    config = SimulationConfig.from_yaml("simulations/ex3_1000_steps_cumulative_stretching_pos2fixed/config.yaml")
    simulation = MolecularDynamics(polyisoprene, config)
    simulation.run()

if __name__ == '__main__':
    
    exercise_1()
    
    # simulation_path = "simulations/ex3_1000_steps_stretching"
    # analyze_distances(simulation_path)
    # analyze_stretches(simulation_path)