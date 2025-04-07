import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mapfbench.core.environment import Environment
from mapfbench.core.benchmark import Benchmark
from mapfbench.algorithms import ConflictBasedSearch, DecentralizedPathfinding

def main():
    # Create a 3D environment
    env_size = (50, 50, 50)
    env = Environment(env_size, obstacle_density=0.1, terrain_complexity=0.2)
    
    # Run CBS Benchmark
    cbs_benchmark = Benchmark(
        environment=env, 
        algorithm_class=ConflictBasedSearch, 
        num_agents=10
    )
    cbs_benchmark.run()
    cbs_benchmark.print_results()
    
    print("\n" + "="*50 + "\n")
    
    # Run Decentralized Benchmark
    decentralized_benchmark = Benchmark(
        environment=env, 
        algorithm_class=DecentralizedPathfinding, 
        num_agents=10
    )
    decentralized_benchmark.run()
    decentralized_benchmark.print_results()

if __name__ == "__main__":
    main()
