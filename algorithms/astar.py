import heapq
import numpy as np
from typing import List, Tuple, Optional

from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class AStar(BaseAlgorithm):
    """
    A* Pathfinding Algorithm implementation.
    
    Uses a heuristic-based search to find optimal paths.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None,
                 heuristic_weight: float = 1.0):
        """
        Initialize A* algorithm.
        
        Args:
            environment: Environment for pathfinding (optional)
            agents: List of agents to find paths for (optional)
            heuristic_weight: Weight for heuristic estimation (default 1.0)
        """
        super().__init__(environment, agents)
        
        # Heuristic weight parameter
        self.heuristic_weight = heuristic_weight
        
        # Collision checks tracking
        self.collision_checks = 0
    
    def _heuristic(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """
        Calculate heuristic distance between two points.
        
        Args:
            a: First point
            b: Second point
        
        Returns:
            Estimated distance between points
        """
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def _reconstruct_path(self, came_from, current):
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
        Find path from start to goal using A* algorithm.
        
        Args:
            start: Starting position
            goal: Goal position
        
        Returns:
            Path from start to goal
        """
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            # Check if goal reached
            if self._heuristic(current, goal) < self.environment.constraints.get('collision_radius', 1.0):
                return self._reconstruct_path(came_from, current)
            
            # Get neighbors
            neighbors = self.environment.get_neighbors(current, goal)
            
            for neighbor in neighbors:
                # Collision check
                self.collision_checks += 1
                
                # Tentative g_score
                tentative_g_score = g_score[current] + self._heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path is better than previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def solve(self):
        """
        Solve paths for all agents.
        """
        for agent in self.agents:
            path = self._find_path(agent.start, agent.goal)
            agent.path = path
            agent.update_position(path[-1] if path else agent.start)
