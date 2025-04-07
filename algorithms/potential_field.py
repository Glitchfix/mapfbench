import numpy as np
from typing import List, Tuple, Optional

from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class PotentialField(BaseAlgorithm):
    """
    Potential Field Pathfinding Algorithm.
    
    Uses attractive and repulsive forces for path planning.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None,
                 attractive_gain: float = 1.0,
                 repulsive_gain: float = 1.0,
                 influence_radius: float = 10.0):
        """
        Initialize Potential Field algorithm.
        
        Args:
            environment: Environment for pathfinding (optional)
            agents: List of agents to find paths for (optional)
            attractive_gain: Gain for attractive force
            repulsive_gain: Gain for repulsive force
            influence_radius: Distance at which obstacles start repelling
        """
        super().__init__(environment, agents)
        
        # Potential field specific parameters
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_radius = influence_radius
        
        # Collision checks tracking
        self.collision_checks = 0
    
    def _attractive_force(self, 
                           current: Tuple[float, float, float], 
                           goal: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate attractive force towards the goal.
        
        Args:
            current: Current position
            goal: Goal position
        
        Returns:
            Attractive force vector
        """
        return self.attractive_gain * (np.array(goal) - np.array(current))
    
    def _repulsive_force(self, 
                          current: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate cumulative repulsive force from all obstacles.
        
        Args:
            current: Current position
        
        Returns:
            Repulsive force vector
        """
        repulsive_force = np.zeros(3)
        
        for obstacle in self.environment.obstacles:
            # Calculate distance to obstacle
            dist = self._distance_to_obstacle(current, obstacle)
            
            # Check if obstacle is within influence radius
            if dist <= self.influence_radius:
                # Direction away from obstacle
                obstacle_vector = np.array(current) - np.array(obstacle.center)
                
                # Normalize and scale by repulsive gain
                repulsive_force += (
                    self.repulsive_gain * 
                    (1/dist - 1/self.influence_radius) * 
                    (obstacle_vector / dist)
                )
        
        return repulsive_force
    
    def _distance_to_obstacle(self, 
                               point: Tuple[float, float, float], 
                               obstacle) -> float:
        """
        Calculate distance to an obstacle.
        
        Args:
            point: Current position
            obstacle: Obstacle object
        
        Returns:
            Distance to obstacle
        """
        self.collision_checks += 1
        
        if hasattr(obstacle, 'contains') and obstacle.contains(point):
            return 0
        
        # Euclidean distance to obstacle center
        return np.linalg.norm(np.array(point) - np.array(obstacle.center))
    
    def _find_path(self, 
                   start: Tuple[float, float, float], 
                   goal: Tuple[float, float, float], 
                   max_iterations: int = 100) -> List[Tuple[float, float, float]]:
        """
        Find path from start to goal using Potential Field method.
        
        Args:
            start: Starting position
            goal: Goal position
            max_iterations: Maximum iterations to prevent infinite loops
        
        Returns:
            Path from start to goal
        """
        path = [start]
        current = np.array(start)
        
        for _ in range(max_iterations):
            # Check if goal reached
            if np.linalg.norm(current - np.array(goal)) < self.environment.constraints.get('collision_radius', 1.0):
                break
            
            # Calculate forces
            attractive = self._attractive_force(current, goal)
            repulsive = self._repulsive_force(current)
            
            # Combine forces
            total_force = attractive + repulsive
            
            # Move in force direction
            new_point = current + total_force
            
            # Ensure new point is valid
            new_point = tuple(
                max(0, min(new_point[i], self.environment.size[i])) 
                for i in range(3)
            )
            
            # Add to path
            path.append(new_point)
            current = np.array(new_point)
        
        return path
    
    def solve(self):
        """
        Solve paths for all agents.
        """
        for agent in self.agents:
            path = self._find_path(agent.start, agent.goal)
            agent.path = path
            agent.update_position(path[-1] if path else agent.start)
