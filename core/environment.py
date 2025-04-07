import numpy as np
import json
from typing import Tuple, List, Optional, Union, Dict, Any, Type
import random
import itertools

class Obstacle:
    type = 'obstacle'
    """
    Base class for obstacles in the environment.
    """
    def __init__(self, center: Tuple[float, float, float]):
        """
        Initialize an obstacle.
        
        Args:
            center (Tuple[float, float, float]): Center position of the obstacle
        """
        self.center = center
    
    def contains(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is within the obstacle.
        
        Args:
            position (Tuple[float, float, float]): Position to check
        
        Returns:
            bool: True if position is inside the obstacle, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Obstacle':
        """
        Create an obstacle from a dictionary representation.
        
        Args:
            data (Dict[str, Any]): Obstacle configuration
        
        Returns:
            Obstacle: Instantiated obstacle
        """
        obstacle_type = data.get('type', 'sphere')
        obstacle_classes = {
            'sphere': SphereObstacle,
            'cube': CubeObstacle,
            'cylinder': CylinderObstacle
        }
        
        obstacle_class = obstacle_classes.get(obstacle_type.lower())
        if not obstacle_class:
            raise ValueError(f"Unknown obstacle type: {obstacle_type}")
        obs = obstacle_class(**data)
        obs.type = obstacle_type.lower()
        return obs

class SphereObstacle(Obstacle):
    """
    Spherical obstacle with a defined radius.
    """
    def __init__(self, center: Tuple[float, float, float], radius: float, **kwargs):
        """
        Initialize a spherical obstacle.
        
        Args:
            center (Tuple[float, float, float]): Center of the sphere
            radius (float): Radius of the sphere
        """
        super().__init__(center)
        self.radius = radius
    
    def contains(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is inside the sphere.
        
        Args:
            position (Tuple[float, float, float]): Position to check
        
        Returns:
            bool: True if position is inside the sphere, False otherwise
        """
        distance = np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(position, self.center)))
        return distance <= self.radius

class CubeObstacle(Obstacle):
    """
    Cubic obstacle with a defined size.
    """
    def __init__(self, center: Tuple[float, float, float], size: Tuple[float, float, float], **kwargs):
        """
        Initialize a cubic obstacle.
        
        Args:
            center (Tuple[float, float, float]): Center of the cube
            size (Tuple[float, float, float]): Dimensions of the cube
        """
        super().__init__(center)
        self.size = size
    
    def contains(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is inside the cube.
        
        Args:
            position (Tuple[float, float, float]): Position to check
        
        Returns:
            bool: True if position is inside the cube, False otherwise
        """
        return all(
            abs(pos - center) <= size / 2 
            for pos, center, size in zip(position, self.center, self.size)
        )

class CylinderObstacle(Obstacle):
    """
    Cylindrical obstacle with a defined radius and height.
    """
    def __init__(self, center: Tuple[float, float, float], radius: float, height: float, axis: str = 'y', **kwargs):
        """
        Initialize a cylindrical obstacle.
        
        Args:
            center (Tuple[float, float, float]): Center of the cylinder
            radius (float): Radius of the cylinder
            height (float): Height of the cylinder
            axis (str, optional): Cylinder orientation. Defaults to 'y'.
        """
        super().__init__(center)
        self.radius = radius
        self.height = height
        self.axis = axis.lower()
    
    def contains(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is inside the cylinder.
        
        Args:
            position (Tuple[float, float, float]): Position to check
        
        Returns:
            bool: True if position is inside the cylinder, False otherwise
        """
        # Determine indices based on axis
        axis_map = {'x': (1, 2, 0), 'y': (0, 2, 1), 'z': (0, 1, 2)}
        base_idx, height_idx, radius_idx = axis_map[self.axis]
        
        # Check height
        height_diff = abs(position[height_idx] - self.center[height_idx])
        if height_diff > self.height / 2:
            return False
        
        # Check radius
        base_distance = np.sqrt(
            (position[base_idx] - self.center[base_idx])**2 + 
            (position[radius_idx] - self.center[radius_idx])**2
        )
        
        return base_distance <= self.radius

class Environment:
    """
    Flexible 3D environment representation for multiagent path finding.
    
    Supports dynamic obstacles, configurable constraints, and movement rules.
    """
    
    def __init__(self, 
                 size: Tuple[int, int, int] = (100, 100, 100), 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a 3D environment.
        
        Args:
            size (Tuple[int, int, int]): Dimensions of the environment
            config (Optional[Dict[str, Any]]): Configuration parameters for the environment
        """
        self.size = size
        self.config = config or {}
        
        # Environment name
        self.name = self.config.get('name', 'Unnamed Environment')
        
        # Environment constraints
        self.constraints = {
            'max_velocity': self.config.get('max_velocity', 1),
            'collision_radius': self.config.get('collision_radius', 1.0),
            'boundary_behavior': self.config.get('boundary_behavior', 'reflect')
        }
        
        # Obstacles management
        self.obstacles = []
        
        # Load obstacles if provided in config
        if 'obstacles' in self.config:
            self.load_obstacles(self.config['obstacles'])
        
        # Optional dynamic obstacles or no-go zones
        self.dynamic_obstacles = self.config.get('dynamic_obstacles', [])
    
    def load_obstacles(self, obstacles: Union[List[Dict[str, Any]], str]):
        """
        Load obstacles into the environment.
        
        Args:
            obstacles (Union[List[Dict[str, Any]], str]): 
                - List of obstacle configurations
                - Path to a JSON file containing obstacles
        """
        if isinstance(obstacles, str):
            # Load from JSON file
            with open(obstacles, 'r') as f:
                obstacle_data = json.load(f)
            
            # Support multiple obstacle representation formats
            if isinstance(obstacle_data, list):
                obstacles = obstacle_data
            elif 'obstacles' in obstacle_data:
                obstacles = obstacle_data['obstacles']
            else:
                raise ValueError("Invalid obstacle JSON format")
        
        # Create obstacle objects
        self.obstacles = [Obstacle.from_dict(obs_config) for obs_config in obstacles]
    
    def is_obstacle(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is an obstacle.
        
        Args:
            position (Tuple[float, float, float]): Position to check
        
        Returns:
            bool: True if position is an obstacle, False otherwise
        """
        return any(
            obstacle.contains(position) 
            for obstacle in self.obstacles
        )
    
    def is_valid_position(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is within the environment bounds and not an obstacle.
        
        Args:
            position (Tuple[float, float, float]): Position to check
        
        Returns:
            bool: True if position is valid, False otherwise
        """
        return (
            all(0 <= pos < dim for pos, dim in zip(position, self.size)) and
            not self.is_obstacle(position)
        )
    
    def apply_boundary_constraints(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Apply boundary constraints based on configured behavior.
        
        Args:
            position (Tuple[float, float, float]): Position to constrain
        
        Returns:
            Tuple[float, float, float]: Constrained position
        """
        constrained_pos = list(position)
        
        for i in range(3):
            if position[i] < 0 or position[i] >= self.size[i]:
                if self.constraints['boundary_behavior'] == 'reflect':
                    # Reflect off boundaries
                    constrained_pos[i] = max(0, min(position[i], self.size[i] - 1))
                elif self.constraints['boundary_behavior'] == 'wrap':
                    # Wrap around boundaries
                    constrained_pos[i] = position[i] % self.size[i]
                else:
                    # Clip to boundaries
                    constrained_pos[i] = max(0, min(position[i], self.size[i] - 1))
        
        return tuple(constrained_pos)
    
    def check_collision(self, 
                        pos1: Tuple[float, float, float], 
                        pos2: Tuple[float, float, float]) -> bool:
        """
        Check for collision between two positions.
        
        Args:
            pos1 (Tuple[float, float, float]): First position
            pos2 (Tuple[float, float, float]): Second position
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        # Euclidean distance
        distance = np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2)))
        
        # Check against collision radius
        return distance < self.constraints['collision_radius']
    
    def get_possible_moves(self, 
                            current_pos: Tuple[float, float, float], 
                            goal: Optional[Tuple[float, float, float]] = None) -> List[Tuple[float, float, float]]:
        """
        Generate possible moves from current position.
        
        Args:
            current_pos (Tuple[float, float, float]): Current position
            goal (Optional[Tuple[float, float, float]]): Goal position for heuristic-based moves
        
        Returns:
            List[Tuple[float, float, float]]: Possible moves
        """
        moves = []
        max_vel = self.constraints['max_velocity']
        
        # Generate moves within velocity constraints
        for dx in np.linspace(-max_vel, max_vel, 9):
            for dy in np.linspace(-max_vel, max_vel, 9):
                for dz in np.linspace(-max_vel, max_vel, 9):
                    new_pos = (
                        current_pos[0] + dx,
                        current_pos[1] + dy,
                        current_pos[2] + dz
                    )
                    
                    # Apply boundary constraints and obstacle check
                    new_pos = self.apply_boundary_constraints(new_pos)
                    
                    if self.is_valid_position(new_pos):
                        moves.append(new_pos)
        
        return moves
    
    def get_neighbors(self, 
                        current_pos: Tuple[float, float, float], 
                        goal: Optional[Tuple[float, float, float]] = None) -> List[Tuple[float, float, float]]:
        """
        Get valid neighboring positions from the current position.
        
        Args:
            current_pos (Tuple[float, float, float]): Current position
            goal (Optional[Tuple[float, float, float]]): Goal position for heuristic-based moves
        
        Returns:
            List[Tuple[float, float, float]]: List of valid neighboring positions
        """
        # Define possible move directions (3D movement)
        directions = [
            (1, 0, 0), (-1, 0, 0),  # X-axis
            (0, 1, 0), (0, -1, 0),  # Y-axis
            (0, 0, 1), (0, 0, -1),  # Z-axis
            
            # Diagonal moves
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
        ]
        
        # Maximum velocity from constraints
        max_velocity = self.constraints.get('max_velocity', 1)
        
        neighbors = []
        for direction in directions:
            # Scale move by max velocity
            move = tuple(max_velocity * d for d in direction)
            
            # Calculate new position
            new_pos = tuple(
                current_pos[i] + move[i] 
                for i in range(len(current_pos))
            )
            
            # Check if the new position is valid
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        # Optional: Sort neighbors by proximity to goal if provided
        if goal:
            neighbors.sort(
                key=lambda pos: sum(
                    abs(pos[i] - goal[i]) 
                    for i in range(len(pos))
                )
            )
        
        return neighbors
    
    def save_to_json(self, file_path: str):
        """
        Save environment configuration to a JSON file.
        
        Args:
            file_path (str): Path to save the JSON file
        """
        json_data = {
            'name': self.name,
            'size': self.size,
            'constraints': self.constraints,
            'obstacles': [
                {
                    'type': type(obs).__name__.replace('Obstacle', '').lower(),
                    'center': obs.center,
                    **{k: v for k, v in vars(obs).items() if k not in ['center']}
                } 
                for obs in self.obstacles
            ],
            'dynamic_obstacles': self.dynamic_obstacles
        }
        
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, file_path: str) -> 'Environment':
        """
        Load environment configuration from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file
        
        Returns:
            Environment: Configured environment instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            size=tuple(data.get('size', (100, 100, 100))),
            config={
                'name': data.get('name', 'Unnamed Environment'),
                'max_velocity': data.get('constraints', {}).get('max_velocity', 1),
                'collision_radius': data.get('constraints', {}).get('collision_radius', 1.0),
                'boundary_behavior': data.get('constraints', {}).get('boundary_behavior', 'reflect'),
                'obstacles': data.get('obstacles', []),
                'dynamic_obstacles': data.get('dynamic_obstacles', [])
            }
        )
    
    def generate_random_position(self) -> Tuple[float, float, float]:
        """
        Generate a random position within the environment.
        
        Returns:
            Tuple[float, float, float]: Random position
        """
        while True:
            pos = tuple(
                random.uniform(0, dim) 
                for dim in self.size
            )
            
            # Ensure the position is valid (not an obstacle)
            if self.is_valid_position(pos):
                return pos
