from typing import List, Tuple, Dict, Any
import random
import time

# Use absolute imports
from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class DecentralizedPathfinding(BaseAlgorithm):
    """
    Decentralized multiagent path finding algorithm.
    
    Agents plan paths independently with local collision avoidance.
    """
    
    def solve(self):
        """
        Solve multiagent path finding using a decentralized approach.
        
        Core steps:
        1. Compute initial paths for all agents
        2. Detect and resolve local conflicts
        3. Update agent paths
        """
        max_iterations = self.config.get('max_iterations', 100)
        
        # Compute initial paths
        for agent in self.agents:
            path = self._compute_path_with_local_avoidance(agent)
            agent.path = path
            agent.update_position(path[-1])
    
    def _compute_path_with_local_avoidance(self, agent: Agent) -> List[Tuple[int, int, int]]:
        """
        Compute a path for an agent with local collision avoidance.
        
        Args:
            agent (Agent): Agent to compute path for
        
        Returns:
            List[Tuple[int, int, int]]: Computed path
        """
        path = [agent.start]
        current = agent.start
        
        while current != agent.goal:
            # Get possible moves
            possible_moves = self._get_possible_moves(current, agent)
            
            # Filter out moves that would cause collisions
            safe_moves = self._filter_safe_moves(possible_moves, path, agent)
            
            if not safe_moves:
                # If no safe moves, wait or backtrack
                path.append(current)
                continue
            
            # Choose move closest to goal with some randomness
            next_move = self._choose_move(safe_moves, agent.goal)
            
            path.append(next_move)
            current = next_move
            
            # Prevent infinite loops
            if len(path) > 100 or len(path) > self._max_possible_path_length(agent):
                break
        
        return path
    
    def _get_possible_moves(self, 
                             current: Tuple[int, int, int], 
                             agent: Agent) -> List[Tuple[int, int, int]]:
        """
        Get possible moves from current position.
        
        Args:
            current (Tuple[int, int, int]): Current position
            agent (Agent): Agent making the move
        
        Returns:
            List[Tuple[int, int, int]]: Possible moves
        """
        moves = self.environment.get_neighbors(current)
        
        # Add wait option (staying in place)
        moves.append(current)
        
        return moves
    
    def _filter_safe_moves(self, 
                            moves: List[Tuple[int, int, int]], 
                            current_path: List[Tuple[int, int, int]], 
                            agent: Agent) -> List[Tuple[int, int, int]]:
        """
        Filter moves to avoid collisions with other agents.
        
        Args:
            moves (List[Tuple[int, int, int]]): Possible moves
            current_path (List[Tuple[int, int, int]]): Current agent's path
            agent (Agent): Agent making the move
        
        Returns:
            List[Tuple[int, int, int]]: Safe moves
        """
        safe_moves = []
        
        for move in moves:
            # Check if move is safe from other agents
            is_safe = all(
                not self._is_move_conflicting(move, current_path, other_agent) 
                for other_agent in self.agents if other_agent.id != agent.id
            )
            
            if is_safe:
                safe_moves.append(move)
        
        return safe_moves
    
    def _is_move_conflicting(self, 
                              move: Tuple[int, int, int], 
                              current_path: List[Tuple[int, int, int]], 
                              other_agent: Agent) -> bool:
        """
        Check if a move conflicts with another agent's path.
        
        Args:
            move (Tuple[int, int, int]): Proposed move
            current_path (List[Tuple[int, int, int]]): Current agent's path
            other_agent (Agent): Other agent to check for conflicts
        
        Returns:
            bool: True if move conflicts, False otherwise
        """
        # Check vertex and edge conflicts
        path_length = len(current_path)
        
        for t, other_path_pos in enumerate(other_agent.path):
            # Vertex conflict
            if move == other_path_pos:
                return True
            
            # Edge conflict (swap positions)
            if (t > 0 and 
                path_length > t and 
                move == other_agent.path[t-1] and 
                current_path[t] == other_path_pos):
                return True
        
        return False
    
    def _choose_move(self, 
                     moves: List[Tuple[int, int, int]], 
                     goal: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Choose a move based on proximity to goal with some randomness.
        
        Args:
            moves (List[Tuple[int, int, int]]): Possible moves
            goal (Tuple[int, int, int]): Goal position
        
        Returns:
            Tuple[int, int, int]: Chosen move
        """
        # Add some randomness to path selection
        if random.random() < 0.2:  # 20% chance of random move
            return random.choice(moves)
        
        # Otherwise, choose move closest to goal
        return min(
            moves, 
            key=lambda move: sum(abs(move[i] - goal[i]) for i in range(3))
        )
    
    def _max_possible_path_length(self, agent: Agent) -> int:
        """
        Compute maximum possible path length based on environment size.
        
        Args:
            agent (Agent): Agent to compute max path length for
        
        Returns:
            int: Maximum possible path length
        """
        return sum(self.environment.size) * 2  # Conservative estimate
