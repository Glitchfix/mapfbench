import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict

from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class Dijkstra(BaseAlgorithm):
    """
    Dijkstra's Pathfinding Algorithm implementation.
    
    Finds shortest paths using uniform cost search.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None,
                 distance_metric: str = 'euclidean'):
        """
        Initialize Dijkstra algorithm.
        
        Args:
            environment: Environment for pathfinding (optional)
            agents: List of agents to find paths for (optional)
            distance_metric: Distance calculation method (default 'euclidean')
        """
        super().__init__(environment, agents)
        
        # Distance metric parameter
        self.distance_metric = distance_metric
        
        # Collision checks tracking
        self.collision_checks = 0
    
    def _distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            a: First point
            b: Second point
        
        Returns:
            Distance between points
        """
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Reconstruct path from start to goal.
        
        Args:
            came_from: Dictionary tracking path
            current: Current node
        
        Returns:
            Reconstructed path
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))
    
    def _find_path(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Find path from start to goal using Dijkstra's algorithm.
        
        Args:
            start: Starting position
            goal: Goal position
        
        Returns:
            Path from start to goal
        """
        # Priority queue for nodes to explore
        pq = [(0, start)]
        
        # Track best known distances
        distances = {start: 0}
        came_from = {}
        
        while pq:
            current_distance, current = heapq.heappop(pq)
            
            # Check if goal reached
            if self._distance(current, goal) < self.environment.constraints.get('collision_radius', 1.0):
                return self._reconstruct_path(came_from, current)
            
            # Skip if we've found a better path
            if current_distance > distances.get(current, float('inf')):
                continue
            
            # Get neighbors
            neighbors = self.environment.get_neighbors(current, goal)
            
            for neighbor in neighbors:
                # Collision check
                self.collision_checks += 1
                
                # Calculate distance to neighbor
                distance = current_distance + self._distance(current, neighbor)
                
                # Update if we've found a better path
                if distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = distance
                    came_from[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return []  # No path found
    
    def solve(self):
        """
        Solve paths for all agents.
        """
        for agent in self.agents:
            path = self._find_path(agent.start, agent.goal)
            agent.path = path
            agent.update_position(path[-1] if path else agent.start)
