import numpy as np
from typing import List, Tuple, Optional, Dict

from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class RRTStar(BaseAlgorithm):
    """
    RRT* (Rapidly-exploring Random Tree Star) Pathfinding Algorithm.
    
    Provides optimal path planning in complex environments.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None,
                 max_iterations: int = 1000,
                 step_size: float = 1.0):
        """
        Initialize RRT* algorithm.
        
        Args:
            environment: Environment for pathfinding (optional)
            agents: List of agents to find paths for (optional)
            max_iterations: Maximum number of iterations for tree expansion
            step_size: Step size for tree growth
        """
        super().__init__(environment, agents)
        
        # RRT* specific parameters
        self.max_iterations = max_iterations
        self.step_size = step_size
        
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
    
    def _steer(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Steer from start towards goal with a maximum step size.
        
        Args:
            start: Starting point
            goal: Goal point
        
        Returns:
            New point in the direction of goal
        """
        direction = np.array(goal) - np.array(start)
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return tuple(goal)
        
        unit_direction = direction / distance
        new_point = np.array(start) + unit_direction * self.step_size
        return tuple(new_point)
    
    def _find_nearest_node(self, tree: List[Tuple[float, float, float]], point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Find the nearest node in the tree to a given point.
        
        Args:
            tree: List of nodes in the tree
            point: Target point
        
        Returns:
            Nearest node in the tree
        """
        return min(tree, key=lambda node: self._distance(node, point))
    
    def _find_path(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Find path from start to goal using RRT* algorithm.
        
        Args:
            start: Starting position
            goal: Goal position
        
        Returns:
            Path from start to goal
        """
        tree = [start]
        tree_edges = {}
        
        for _ in range(self.max_iterations):
            # Sample random point
            if np.random.random() < 0.1:  # Bias towards goal
                random_point = goal
            else:
                random_point = tuple(
                    np.random.uniform(0, self.environment.size[i]) 
                    for i in range(3)
                )
            
            # Find nearest node
            nearest_node = self._find_nearest_node(tree, random_point)
            
            # Steer towards random point
            new_point = self._steer(nearest_node, random_point)
            
            # Collision check
            self.collision_checks += 1
            
            # Check if path is valid
            if self.environment.is_valid_position(new_point):
                # Find near nodes for rewiring
                near_nodes = [
                    node for node in tree 
                    if self._distance(node, new_point) < self.step_size * 2
                ]
                
                # Choose parent with minimum cost
                min_cost_parent = nearest_node
                min_cost = self._distance(start, nearest_node) + self._distance(nearest_node, new_point)
                
                for near_node in near_nodes:
                    potential_cost = (
                        self._distance(start, near_node) + 
                        self._distance(near_node, new_point)
                    )
                    
                    if potential_cost < min_cost:
                        min_cost_parent = near_node
                        min_cost = potential_cost
                
                # Add new node
                tree.append(new_point)
                tree_edges[new_point] = min_cost_parent
                
                # Rewire tree
                for near_node in near_nodes:
                    potential_cost = (
                        self._distance(start, new_point) + 
                        self._distance(new_point, near_node)
                    )
                    
                    if potential_cost < self._distance(start, near_node):
                        tree_edges[near_node] = new_point
                
                # Check if goal is reached
                if self._distance(new_point, goal) < self.step_size:
                    # Reconstruct path
                    path = [goal]
                    current = new_point
                    while current != start:
                        path.append(current)
                        current = tree_edges[current]
                    path.append(start)
                    return list(reversed(path))
        
        return []  # No path found
    
    def solve(self):
        """
        Solve paths for all agents.
        """
        for agent in self.agents:
            path = self._find_path(agent.start, agent.goal)
            agent.path = path
            agent.update_position(path[-1] if path else agent.start)
