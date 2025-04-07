from typing import List, Tuple, Dict, Any
import heapq
import time

# Use absolute imports
from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class ConflictBasedSearch(BaseAlgorithm):
    """
    Conflict-Based Search (CBS) algorithm for multiagent path finding.
    
    Implements a hierarchical search approach to find collision-free paths.
    """
    
    def solve(self):
        """
        Solve multiagent path finding using Conflict-Based Search.
        
        Core steps:
        1. Compute initial paths for all agents
        2. Detect and resolve conflicts
        3. Update agent paths
        """
        # Compute initial paths
        paths = [self._compute_single_agent_path(agent) for agent in self.agents]
        
        # Detect and resolve conflicts
        while not self._is_solution_valid(paths):
            conflict = self._find_first_conflict(paths)
            
            if conflict is None:
                break
            
            # Resolve conflict by re-routing agents
            paths = self._resolve_conflict(paths, conflict)
        
        # Update agent paths
        for agent, path in zip(self.agents, paths):
            agent.path = path
            agent.update_position(path[-1])
    
    def _compute_single_agent_path(self, agent: Agent) -> List[Tuple[int, int, int]]:
        """
        Compute a path for a single agent using A* search.
        
        Args:
            agent (Agent): Agent to compute path for
        
        Returns:
            List[Tuple[int, int, int]]: Computed path
        """
        open_set = []
        heapq.heappush(open_set, (0, agent.start, [agent.start]))
        closed_set = set()
        
        while open_set:
            f_cost, current, path = heapq.heappop(open_set)
            
            if current == agent.goal:
                return path
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self.environment.get_neighbors(current):
                if neighbor not in closed_set:
                    new_path = path + [neighbor]
                    g_cost = len(new_path)
                    h_cost = self._compute_heuristic(neighbor, agent.goal)
                    f_cost = g_cost + h_cost
                    
                    heapq.heappush(open_set, (f_cost, neighbor, new_path))
        
        return [agent.start]  # Fallback if no path found
    
    def _is_solution_valid(self, paths: List[List[Tuple[int, int, int]]]) -> bool:
        """
        Check if the current solution is free of conflicts.
        
        Args:
            paths (List[List[Tuple[int, int, int]]]): Paths for all agents
        
        Returns:
            bool: True if no conflicts, False otherwise
        """
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if self._check_collision(paths[i], paths[j]):
                    return False
        return True
    
    def _find_first_conflict(self, 
                              paths: List[List[Tuple[int, int, int]]]) -> Dict[str, Any]:
        """
        Find the first conflict between agent paths.
        
        Args:
            paths (List[List[Tuple[int, int, int]]]): Paths for all agents
        
        Returns:
            Dict[str, Any]: Conflict details or None
        """
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                max_length = max(len(paths[i]), len(paths[j]))
                
                for t in range(max_length):
                    pos1 = paths[i][min(t, len(paths[i]) - 1)]
                    pos2 = paths[j][min(t, len(paths[j]) - 1)]
                    
                    # Vertex collision
                    if pos1 == pos2:
                        return {
                            'agents': (i, j),
                            'time_step': t,
                            'positions': (pos1, pos2),
                            'type': 'vertex'
                        }
                    
                    # Edge collision
                    if t > 0:
                        prev_pos1 = paths[i][min(t-1, len(paths[i]) - 1)]
                        prev_pos2 = paths[j][min(t-1, len(paths[j]) - 1)]
                        
                        if pos1 == prev_pos2 and pos2 == prev_pos1:
                            return {
                                'agents': (i, j),
                                'time_step': t,
                                'positions': (pos1, pos2),
                                'type': 'edge'
                            }
        
        return None
    
    def _resolve_conflict(self, 
                           paths: List[List[Tuple[int, int, int]]], 
                           conflict: Dict[str, Any]) -> List[List[Tuple[int, int, int]]]:
        """
        Resolve a conflict by modifying paths.
        
        Args:
            paths (List[List[Tuple[int, int, int]]]): Current paths
            conflict (Dict[str, Any]): Conflict details
        
        Returns:
            List[List[Tuple[int, int, int]]]: Updated paths
        """
        agent1, agent2 = conflict['agents']
        
        # Create constraints to force path modifications
        constraints = [
            {
                'agent': agent1,
                'position': conflict['positions'][1],
                'time_step': conflict['time_step']
            },
            {
                'agent': agent2,
                'position': conflict['positions'][0],
                'time_step': conflict['time_step']
            }
        ]
        
        # Recompute paths with constraints
        new_paths = paths.copy()
        for constraint in constraints:
            agent_index = constraint['agent']
            new_paths[agent_index] = self._compute_constrained_path(
                self.agents[agent_index], 
                constraint
            )
        
        return new_paths
    
    def _compute_constrained_path(self, 
                                   agent: Agent, 
                                   constraint: Dict[str, Any]) -> List[Tuple[int, int, int]]:
        """
        Compute a path with additional constraints.
        
        Args:
            agent (Agent): Agent to compute path for
            constraint (Dict[str, Any]): Path constraint
        
        Returns:
            List[Tuple[int, int, int]]: Constrained path
        """
        # Similar to _compute_single_agent_path, but with additional constraint check
        open_set = []
        heapq.heappush(open_set, (0, agent.start, [agent.start]))
        closed_set = set()
        
        while open_set:
            f_cost, current, path = heapq.heappop(open_set)
            
            # Check constraint
            if (current == constraint['position'] and 
                len(path) - 1 == constraint['time_step']):
                continue
            
            if current == agent.goal:
                return path
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self.environment.get_neighbors(current):
                if neighbor not in closed_set:
                    new_path = path + [neighbor]
                    g_cost = len(new_path)
                    h_cost = self._compute_heuristic(neighbor, agent.goal)
                    f_cost = g_cost + h_cost
                    
                    heapq.heappush(open_set, (f_cost, neighbor, new_path))
        
        return [agent.start]  # Fallback if no path found
