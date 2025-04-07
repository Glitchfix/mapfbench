from typing import Tuple, List, Optional, Dict, Any
import uuid

class Agent:
    """
    Flexible agent representation for multiagent path finding.
    
    Supports custom attributes, path tracking, and performance metrics.
    """
    
    def __init__(self, 
                 start: Tuple[int, int, int], 
                 goal: Tuple[int, int, int], 
                 attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize an agent with start and goal positions.
        
        Args:
            start (Tuple[int, int, int]): Starting position
            goal (Tuple[int, int, int]): Goal position
            attributes (Optional[Dict[str, Any]]): Additional agent attributes
        """
        self.id = str(uuid.uuid4())  # Unique identifier
        self.start = start
        self.goal = goal
        self.current_position = start
        
        # Path tracking
        self.path: List[Tuple[int, int, int]] = [start]
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            'path_length': 0,
            'time_to_goal': 0,
            'collisions': 0
        }
        
        # Custom attributes
        self.attributes = attributes or {}
    
    def update_position(self, new_position: Tuple[int, int, int]):
        """
        Update agent's current position and path.
        
        Args:
            new_position (Tuple[int, int, int]): New position
        """
        self.current_position = new_position
        self.path.append(new_position)
        self.metrics['path_length'] = len(self.path)
    
    def is_goal_reached(self) -> bool:
        """
        Check if the agent has reached its goal.
        
        Returns:
            bool: True if goal is reached, False otherwise
        """
        return self.current_position == self.goal
    
    def reset(self):
        """
        Reset agent to initial state.
        """
        self.current_position = self.start
        self.path = [self.start]
        self.metrics = {
            'path_length': 0,
            'time_to_goal': 0,
            'collisions': 0
        }
    
    def add_metric(self, key: str, value: Any):
        """
        Add or update a custom metric.
        
        Args:
            key (str): Metric name
            value (Any): Metric value
        """
        self.metrics[key] = value
    
    def get_metric(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a specific metric.
        
        Args:
            key (str): Metric name
            default (Any): Default value if metric doesn't exist
        
        Returns:
            Any: Metric value or default
        """
        return self.metrics.get(key, default)
    
    def __repr__(self) -> str:
        """
        String representation of the agent.
        
        Returns:
            str: Agent details
        """
        return (f"Agent(id={self.id}, "
                f"start={self.start}, "
                f"goal={self.goal}, "
                f"current={self.current_position})")
    
    def __eq__(self, other) -> bool:
        """
        Compare agents based on their unique ID.
        
        Args:
            other (Agent): Another agent to compare
        
        Returns:
            bool: True if agents have the same ID, False otherwise
        """
        if not isinstance(other, Agent):
            return False
        return self.id == other.id
