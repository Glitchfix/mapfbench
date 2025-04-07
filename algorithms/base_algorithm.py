from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

# Use absolute imports
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class BaseAlgorithm(ABC):
    """
    Abstract base class for multiagent path finding algorithms.
    
    Provides a standard interface for implementing path finding strategies.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None, 
                 **kwargs):
        """
        Initialize the algorithm with optional environment and agents.
        
        Args:
            environment (Environment): The environment to navigate (optional)
            agents (List[Agent]): List of agents to plan paths for (optional)
            **kwargs: Additional configuration parameters
        """
        self.environment = environment
        self.agents = agents if agents is not None else []
        self.config = kwargs
    
    def set_environment(self, environment: Environment):
        """
        Set the environment for the algorithm.
        
        Args:
            environment: Environment to set
        """
        self.environment = environment
    
    def set_agents(self, agents: List[Agent]):
        """
        Set the agents for the algorithm.
        
        Args:
            agents: List of agents to set
        """
        self.agents = agents
    
    @abstractmethod
    def solve(self):
        """
        Abstract method to solve the multiagent path finding problem.
        
        Implementations should:
        1. Plan paths for all agents
        2. Update agent paths
        3. Handle potential conflicts
        """
        pass
    
    def _check_collision(self, path1: List[Tuple[int, int, int]], 
                          path2: List[Tuple[int, int, int]]) -> bool:
        """
        Check for collisions between two agent paths.
        
        Args:
            path1 (List[Tuple[int, int, int]]): First agent's path
            path2 (List[Tuple[int, int, int]]): Second agent's path
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        max_length = max(len(path1), len(path2))
        
        for t in range(max_length):
            # Check vertex and edge collisions
            pos1 = path1[min(t, len(path1) - 1)]
            pos2 = path2[min(t, len(path2) - 1)]
            
            # Vertex collision
            if pos1 == pos2:
                return True
            
            # Edge collision (swap positions)
            if t > 0:
                prev_pos1 = path1[min(t-1, len(path1) - 1)]
                prev_pos2 = path2[min(t-1, len(path2) - 1)]
                
                if pos1 == prev_pos2 and pos2 == prev_pos1:
                    return True
        
        return False
    
    def _compute_heuristic(self, start: Tuple[int, int, int], 
                            goal: Tuple[int, int, int]) -> float:
        """
        Compute a heuristic distance between start and goal.
        
        Args:
            start (Tuple[int, int, int]): Starting position
            goal (Tuple[int, int, int]): Goal position
        
        Returns:
            float: Estimated distance
        """
        return sum(abs(s - g) for s, g in zip(start, goal))
    
    def get_algorithm_name(self) -> str:
        """
        Get the name of the current algorithm.
        
        Returns:
            str: Algorithm name
        """
        return self.__class__.__name__
