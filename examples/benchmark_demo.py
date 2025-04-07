import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import os
import csv
from datetime import datetime

from mapfbench.core.environment import Environment
from mapfbench.core.benchmark import Benchmark
from mapfbench.algorithms import (
    AStar, 
    Dijkstra, 
    RRTStar, 
    PotentialField, 
    Bug1, 
    Bug2,
    LSTMBaggingPathfinder,
)

def main():
    """
    Demonstrate benchmarking multiple pathfinding algorithms.
    """
    # Load environment from JSON
    env_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'environments', 
        'cityscape.json'
    )
    env = Environment.load_from_json(env_path)
    num_agents = 4
    random_seed = 44
    # List of algorithms to benchmark
    algorithms = [
        AStar(heuristic_weight=1.0),
        RRTStar(),
        Bug1(),
        Bug2(),
        LSTMBaggingPathfinder(num_models=1, lstm_units=256, epochs=1000),
        Dijkstra(distance_metric='euclidean'),
    ]
    
    # Prepare to collect all metrics
    all_metrics = []
    
    # Create benchmark
    benchmark = Benchmark(
        environment=env, 
        num_agents=num_agents, 
        seed=random_seed  # For reproducibility
    )
    
    # Run benchmarks for each algorithm
    for algorithm in algorithms:
        print(f"\n=== Benchmarking {algorithm.__class__.__name__} ===")
        
        # Use the pre-initialized algorithm
        benchmark.use_algorithm(algorithm)
        
        # Run benchmark
        benchmark.run()
        
        # Print results
        benchmark.print_results()
        
        # Visualize paths
        # benchmark.visualize(mode='static')
        benchmark.visualize(mode='animate', save_path='animation')
        
        # Collect metrics
        algorithm_metrics = {
            'algorithm_name': type(algorithm).__name__,
            'num_agents': num_agents,
            'environment': os.path.basename(env_path),
            'seed': random_seed
        }
        
        # Add benchmark metrics
        algorithm_metrics.update(benchmark.metrics)
        del algorithm_metrics['paths']
        
        # Add to overall metrics
        all_metrics.append(algorithm_metrics)
        
        # Optional: dump individual algorithm metrics
        # json_metrics_path = benchmark.dump_metrics(format='json')
        # print(f"Metrics saved to JSON: {json_metrics_path}")
    
    
    
    # Prepare output directory
    output_dir = os.path.join(os.getcwd(), 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_metrics_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Collect all unique keys
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(list(all_keys)))
        writer.writeheader()
        
        for metrics in all_metrics:
            writer.writerow(metrics)
    
    print(f"Comprehensive metrics saved to CSV: {csv_path}")

if __name__ == '__main__':
    main()
