from typing import List, Tuple, Optional
import numpy as np

from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class Bug1(BaseAlgorithm):
    """
    Bug1 Boundary Following Pathfinding Algorithm.
    
    Navigates around obstacles by following their boundaries.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None,
                 hit_point_threshold: float = 0.1):
        """
        Initialize Bug1 algorithm.
        
        Args:
            environment: Environment for pathfinding (optional)
            agents: List of agents to find paths for (optional)
            hit_point_threshold: Threshold for considering a point as a hit point
        """
        super().__init__(environment, agents)
        
        # Bug1 specific parameters
        self.hit_point_threshold = hit_point_threshold
        
        # Collision checks tracking
        self.collision_checks = 0
    
    def _distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            a: First point coordinates
            b: Second point coordinates
        
        Returns:
            Distance between points
        """
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def solve(self):
        """
        Solve paths for all agents using Bug1 algorithm.
        """
        for agent in self.agents:
            # Reset collision checks
            self.collision_checks = 0
            
            # Initial path is direct line to goal
            path = [agent.start]
            current = agent.start
            
            # Collision radius for determining proximity to goal
            collision_radius = self.environment.constraints.get('collision_radius', 1.0)
            
            # Maximum iterations to prevent infinite loop
            max_iterations = 1000
            iteration = 0
            
            while (self._distance(current, agent.goal) > collision_radius and 
                   iteration < max_iterations):
                iteration += 1
                
                # Collision check
                self.collision_checks += 1
                
                # Direct line towards goal
                direction = np.array(agent.goal) - np.array(current)
                direction_norm = direction / np.linalg.norm(direction)
                
                # Potential next position
                next_pos = tuple(current[i] + direction_norm[i] for i in range(3))
                
                # Check if next position is valid
                if not self.environment.is_valid_position(next_pos):
                    # If invalid, try alternative navigation strategies
                    # 1. Move in different directions to avoid obstacle
                    directions = [
                        (1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1)
                    ]
                    
                    found_valid_point = False
                    for dx, dy, dz in directions:
                        alternative_pos = tuple(
                            current[i] + dx * collision_radius * 0.5 + 
                            dy * collision_radius * 0.5 + 
                            dz * collision_radius * 0.5 
                            for i in range(3)
                        )
                        
                        if self.environment.is_valid_position(alternative_pos):
                            next_pos = alternative_pos
                            found_valid_point = True
                            break
                    
                    # If no valid point found, break to prevent infinite loop
                    if not found_valid_point:
                        break
                
                # Update path
                path.append(next_pos)
                current = next_pos
            
            # Add goal to path
            path.append(agent.goal)
            
            # Update agent's path
            agent.path = path
            
            # Update agent's final position
            agent.update_position(agent.path[-1])

class Bug2(BaseAlgorithm):
    """
    Bug2 Boundary Following Pathfinding Algorithm.
    
    More efficient boundary following method compared to Bug1.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None,
                 m_line_threshold: float = 0.1):
        """
        Initialize Bug2 algorithm.
        
        Args:
            environment: Environment for pathfinding (optional)
            agents: List of agents to find paths for (optional)
            m_line_threshold: Threshold for considering a point on the m-line
        """
        super().__init__(environment, agents)
        
        # Bug2 specific parameters
        self.m_line_threshold = m_line_threshold
        
        # Collision checks tracking
        self.collision_checks = 0
    
    def _distance(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            a: First point coordinates
            b: Second point coordinates
        
        Returns:
            Distance between points
        """
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def _is_on_m_line(self, point: Tuple[float, float, float], start: Tuple[float, float, float], goal: Tuple[float, float, float]) -> bool:
        """
        Check if a point is on the m-line between start and goal.
        
        Args:
            point: Point to check
            start: Starting point
            goal: Goal point
        
        Returns:
            True if point is on m-line, False otherwise
        """
        # Calculate vector from start to goal
        m_line_vector = np.array(goal) - np.array(start)
        
        # Calculate vector from start to point
        point_vector = np.array(point) - np.array(start)
        
        # Project point onto m-line
        projection = np.dot(point_vector, m_line_vector) / np.dot(m_line_vector, m_line_vector)
        
        # Check if projection is close to m-line
        return abs(projection) <= 1.0 and projection >= 0
    
    def solve(self):
        """
        Solve paths for all agents using Bug2 algorithm.
        """
        for agent in self.agents:
            # Reset collision checks
            self.collision_checks = 0
            
            # Initial path is direct line to goal
            path = [agent.start]
            current = agent.start
            
            # Track leave point for return to m-line
            leave_point = None
            
            # Collision radius for determining proximity to goal
            collision_radius = self.environment.constraints.get('collision_radius', 1.0)
            
            # Maximum iterations to prevent infinite loop
            max_iterations = 1000
            iteration = 0
            
            while (self._distance(current, agent.goal) > collision_radius and 
                   iteration < max_iterations):
                iteration += 1
                
                # Collision check
                self.collision_checks += 1
                
                # Direct line towards goal
                direction = np.array(agent.goal) - np.array(current)
                direction_norm = direction / np.linalg.norm(direction)
                
                # Potential next position
                next_pos = tuple(current[i] + direction_norm[i] for i in range(3))
                
                # Check if next position is valid
                if not self.environment.is_valid_position(next_pos):
                    # If first time leaving m-line, record leave point
                    if leave_point is None:
                        leave_point = current
                    
                    # If invalid, try alternative navigation strategies
                    # 1. Move in different directions to avoid obstacle
                    directions = [
                        (1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1)
                    ]
                    
                    found_valid_point = False
                    for dx, dy, dz in directions:
                        alternative_pos = tuple(
                            current[i] + dx * collision_radius * 0.5 + 
                            dy * collision_radius * 0.5 + 
                            dz * collision_radius * 0.5 
                            for i in range(3)
                        )
                        
                        if self.environment.is_valid_position(alternative_pos):
                            next_pos = alternative_pos
                            found_valid_point = True
                            
                            # Check if back on m-line
                            if self._is_on_m_line(next_pos, agent.start, agent.goal):
                                # Reset leave point
                                leave_point = None
                            
                            break
                    
                    # If no valid point found, break to prevent infinite loop
                    if not found_valid_point:
                        break
                
                # Update path
                path.append(next_pos)
                current = next_pos
            
            # Add goal to path
            path.append(agent.goal)
            
            # Update agent's path
            agent.path = path
            
            # Update agent's final position
            agent.update_position(agent.path[-1])