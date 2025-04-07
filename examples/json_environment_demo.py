import os
import sys

# Ensure the parent directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from mapfbench import Environment, Benchmark
from mapfbench import ConflictBasedSearch, DecentralizedPathfinding

def main():
    # Path to the sample JSON environment
    json_path = os.path.join(os.path.dirname(__file__), 'sample_environment.json')
    
    # Load environment from JSON
    env = Environment(json_path)
    
    # Print environment details
    print("Environment loaded from JSON:")
    print(f"Environment size: {env.size}")
    print("Environment grid:")
    print(env.grid)
    
    # Optional: Save the environment to a new JSON file
    output_path = os.path.join(os.path.dirname(__file__), 'saved_environment.json')
    env.save_to_json(output_path)
    print(f"\nEnvironment also saved to: {output_path}")
    
    # Run benchmark with the loaded environment
    benchmark = Benchmark(
        environment=env, 
        algorithm_class=ConflictBasedSearch, 
        num_agents=5
    )
    
    benchmark.run()
    benchmark.print_results()

if __name__ == "__main__":
    main()
