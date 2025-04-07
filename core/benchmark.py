import time
import json
import csv
import os
from datetime import datetime
import numpy as np
from typing import List, Tuple, Type, Dict, Any, Optional
import psutil
import threading

from .environment import Environment
from .agent import Agent
from ..visualization.path_visualizer import PathVisualizer

class BenchmarkResourceMonitor:
    """
    Resource monitoring utility for benchmarking.
    """
    def __init__(self):
        """
        Initialize resource monitor.
        """
        self.cpu_percentages = []
        self.memory_usages = []
        self.stop_event = threading.Event()
        self.monitoring_thread = None
    
    def _monitor(self, interval: float = 0.1):
        """
        Background thread to monitor system resources.
        
        Args:
            interval: Sampling interval in seconds
        """
        while not self.stop_event.is_set():
            # Current process
            process = psutil.Process(os.getpid())
            
            # CPU and memory usage
            self.cpu_percentages.append(process.cpu_percent())
            self.memory_usages.append(process.memory_info().rss / (1024 * 1024))  # MB
            
            # Sleep for interval
            time.sleep(interval)
    
    def start(self):
        """
        Start resource monitoring.
        """
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor)
        self.monitoring_thread.start()
    
    def stop(self):
        """
        Stop resource monitoring.
        
        Returns:
            Dict of resource usage metrics
        """
        self.stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        return {
            'avg_cpu_usage': np.mean(self.cpu_percentages) if self.cpu_percentages else 0,
            'max_cpu_usage': np.max(self.cpu_percentages) if self.cpu_percentages else 0,
            'avg_memory_usage': np.mean(self.memory_usages) if self.memory_usages else 0,
            'max_memory_usage': np.max(self.memory_usages) if self.memory_usages else 0
        }

class Benchmark:
    """
    Benchmark class for multiagent pathfinding algorithms.
    
    Provides performance metrics and visualization for path finding.
    """
    
    def __init__(self, 
                 environment: Environment, 
                 num_agents: int = 10,
                 seed: Optional[int] = None):
        """
        Initialize the benchmark.
        
        Args:
            environment: The environment to run pathfinding in
            num_agents: Number of agents to simulate
            seed: Optional random seed for reproducibility
        """
        # Resource monitor
        self.resource_monitor = BenchmarkResourceMonitor()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        self.environment = environment
        self.num_agents = num_agents
        
        # Generate random start and goal positions (fixed for all algorithm runs)
        self.start_positions = self._generate_positions()
        self.goal_positions = self._generate_positions()
        
        # Algorithm configuration
        self.algorithm_class = None
        self.algorithm_instance = None
        
        # Metrics storage
        self.metrics = {
            'algorithm_name': None,
            'total_computation_time': 0,
            'paths': [],
            'path_lengths': [],
            'collision_checks': 0,
            'total_steps': 0
        }
    
    def _generate_positions(self) -> List[Tuple[float, float, float]]:
        """
        Generate random valid positions within the environment.
        
        Returns:
            List of positions for agents
        """
        positions = []
        for _ in range(self.num_agents):
            while True:
                pos = [
                    np.random.uniform(0, self.environment.size[0]),
                    np.random.uniform(0, self.environment.size[1]),
                    np.random.uniform(0, self.environment.size[2])
                ]
                
                # Check if position is valid (not in an obstacle)
                if not self.environment.is_obstacle(pos):
                    positions.append(tuple(pos))
                    break
        
        return positions
    
    def use_algorithm(self, algorithm_class_or_instance):
        """
        Set the algorithm for benchmarking.
        
        Args:
            algorithm_class_or_instance: Algorithm class or pre-initialized instance
        """
        if isinstance(algorithm_class_or_instance, type):
            # If a class is passed, store the class
            self.algorithm_class = algorithm_class_or_instance
            self.algorithm_instance = None
        else:
            # If an instance is passed, store the instance
            self.algorithm_class = type(algorithm_class_or_instance)
            self.algorithm_instance = algorithm_class_or_instance
        
        # Reset metrics for new algorithm
        self.metrics = {
            'algorithm_name': self.algorithm_class.__name__,
            'total_computation_time': 0,
            'paths': [],
            'path_lengths': [],
            'collision_checks': 0,
            'total_steps': 0
        }
    
    def run(self):
        """
        Run the pathfinding algorithm and collect metrics.
        
        Raises:
            ValueError: If no algorithm has been set
        """
        if self.algorithm_class is None and self.algorithm_instance is None:
            raise ValueError("No algorithm set. Use use_algorithm() first.")
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Create algorithm instance if not already provided
        if self.algorithm_instance is None:
            algorithm = self.algorithm_class()
        else:
            algorithm = self.algorithm_instance
        
        # Create agents with predefined start and goal positions
        agents = [
            Agent(
                start=start, 
                goal=goal, 
                attributes={'environment': self.environment}
            ) 
            for start, goal in zip(self.start_positions, self.goal_positions)
        ]
        
        # Set environment and agents
        algorithm.set_environment(self.environment)
        algorithm.set_agents(agents)
        
        # Measure computation time
        start_time = time.time()
        
        # Solve paths
        algorithm.solve()
        
        # Stop resource monitoring
        resource_metrics = self.resource_monitor.stop()
        
        # Calculate metrics
        end_time = time.time()
        self.metrics['total_computation_time'] = end_time - start_time
        
        # Merge resource metrics
        self.metrics.update(resource_metrics)
        
        # Extract paths from agents
        paths = [agent.path for agent in agents]
        self.metrics['paths'] = paths
        
        # Calculate path lengths and total steps
        path_lengths = []
        total_steps = 0
        
        for path in paths:
            # Euclidean path length
            path_length = sum(
                np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                for i in range(len(path)-1)
            )
            path_lengths.append(path_length)
            
            # Total steps
            total_steps += len(path)
        
        self.metrics['path_lengths'] = path_lengths
        self.metrics['total_steps'] = total_steps
        
        # Collision checks (if supported by algorithm)
        if hasattr(algorithm, 'collision_checks'):
            self.metrics['collision_checks'] = algorithm.collision_checks
    
    def print_results(self):
        """
        Print benchmark results and metrics.
        """
        print(f"\n=== Benchmark Results for {self.metrics['algorithm_name']} ===")
        print(f"Environment: {self.environment.name}")
        print(f"Number of Agents: {self.num_agents}")
        
        # Computation time
        print(f"Total Computation Time: {self.metrics['total_computation_time']:.4f} seconds")
        
        # Resource usage
        print("\nResource Usage:")
        print(f"Average CPU Usage: {self.metrics['avg_cpu_usage']:.2f}%")
        print(f"Max CPU Usage: {self.metrics['max_cpu_usage']:.2f}%")
        print(f"Average Memory Usage: {self.metrics['avg_memory_usage']:.2f} MB")
        print(f"Max Memory Usage: {self.metrics['max_memory_usage']:.2f} MB")
        
        # Path metrics
        print("\nPath Metrics:")
        print(f"Total Path Steps: {self.metrics['total_steps']}")
        print(f"Average Path Length: {np.mean(self.metrics['path_lengths']):.2f}")
        print(f"Max Path Length: {max(self.metrics['path_lengths']):.2f}")
        print(f"Min Path Length: {min(self.metrics['path_lengths']):.2f}")
        
        # Collision checks
        print(f"Total Collision Checks: {self.metrics['collision_checks']}")
    
    def visualize(self, mode='static', save_path=None):
        """
        Visualize the paths.
        
        Args:
            mode: 'static' or 'animate'
            save_path: Optional path to save animations (only used when mode='animate')
        """
        visualizer = PathVisualizer(self.environment)
        
        if mode == 'static':
            visualizer.visualize_paths(
                paths=self.metrics['paths'],
                start_positions=self.start_positions,
                goal_positions=self.goal_positions,
                algorithm_name=self.metrics['algorithm_name']
            )
        elif mode == 'animate':
            visualizer.animate_paths(
                paths=self.metrics['paths'],
                start_positions=self.start_positions,
                goal_positions=self.goal_positions,
                algorithm_name=self.metrics['algorithm_name'],
                save_path=save_path
            )
        else:
            raise ValueError("Mode must be 'static' or 'animate'")

    def export_results(self, format='json'):
        """
        Export benchmark results to a specified format.
        
        Args:
            format: Output format ('json', 'csv', etc.)
        
        Returns:
            Exported results
        """
        if format == 'json':
            import json
            return json.dumps(self.metrics, indent=2)
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write headers
            writer.writerow(self.metrics.keys())
            
            # Write values
            writer.writerow(self.metrics.values())
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def json_serializer(obj):
        """
        Custom JSON serializer to handle NumPy and PyTorch types.
        
        Args:
            obj: Object to serialize
        
        Returns:
            Serializable representation of the object
        """
        import numpy as np
        import torch
        
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def dump_metrics(self, 
                     format: str = 'json', 
                     output_dir: Optional[str] = None, 
                     filename: Optional[str] = None):
        """
        Dump benchmark metrics to a file in specified format.
        
        Args:
            format: Output format, either 'json' or 'csv'
            output_dir: Directory to save metrics. Defaults to 'benchmark_results'
            filename: Custom filename. If None, generates a timestamp-based filename
        
        Returns:
            str: Path to the generated metrics file
        
        Raises:
            ValueError: If an unsupported format is specified
        """
        # Prepare output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'benchmark_results')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.metrics['algorithm_name']}_{timestamp}"
        
        # Collect additional metrics from resource monitor
        resource_metrics = self.resource_monitor.stop()
        
        # Combine metrics
        full_metrics = {
            **self.metrics,
            **resource_metrics,
            'environment_size': self.environment.size,
            'num_agents': self.num_agents,
            'timestamp': datetime.now().isoformat()
        }
        
        # Output based on format
        if format.lower() == 'json':
            filepath = os.path.join(output_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(full_metrics, f, indent=4, default=Benchmark.json_serializer)
        elif format.lower() == 'csv':
            filepath = os.path.join(output_dir, f"{filename}.csv")
            with open(filepath, 'w', newline='') as f:
                # Flatten metrics for CSV, converting non-serializable types
                flattened_metrics = {}
                for key, value in full_metrics.items():
                    if isinstance(value, (list, np.ndarray)):
                        flattened_metrics[key] = str(value)
                    elif isinstance(value, (np.integer, np.floating, float, int)):
                        flattened_metrics[key] = float(value)
                    else:
                        flattened_metrics[key] = value
                
                writer = csv.DictWriter(f, fieldnames=list(flattened_metrics.keys()))
                writer.writeheader()
                writer.writerow(flattened_metrics)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")
        
        return filepath
