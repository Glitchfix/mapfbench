import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Tuple, Dict, Any, Optional

class PathVisualizer:
    """
    3D path visualization and animation for multiagent pathfinding.
    """
    
    def __init__(self, environment):
        """
        Initialize the visualizer with an environment.
        
        Args:
            environment: The environment to visualize
        """
        self.environment = environment
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
    
    def _plot_obstacles(self):
        """
        Plot obstacles in the environment.
        """
        for obstacle in self.environment.obstacles:
            try:
                # Normalize obstacle representation
                if isinstance(obstacle, dict):
                    center = obstacle.get('center', [0, 0, 0])
                    obstacle_type = obstacle.get('type', 'cube').lower()
                    radius = obstacle.get('radius', 1)
                    size = obstacle.get('size', [1, 1, 1])
                else:
                    center = obstacle.center
                    obstacle_type = type(obstacle).__name__.lower().replace('obstacle', '')
                    radius = getattr(obstacle, 'radius', 1)
                    size = getattr(obstacle, 'size', [1, 1, 1])
                
                # Ensure center is a list/tuple
                center = list(center)
                
                if obstacle_type == 'sphere':
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    x = radius * np.cos(u) * np.sin(v) + center[0]
                    y = radius * np.sin(u) * np.sin(v) + center[1]
                    z = radius * np.cos(v) + center[2]
                    self.ax.plot_surface(x, y, z, color='gray', alpha=0.3)
                
                elif obstacle_type == 'cube':
                    # Ensure size is a list
                    size = list(size)
                    if len(size) < 3:
                        size.extend([1] * (3 - len(size)))
                    
                    half_size = [s/2 for s in size]
                    x = [center[0] - half_size[0], center[0] + half_size[0]]
                    y = [center[1] - half_size[1], center[1] + half_size[1]]
                    z = [center[2] - half_size[2], center[2] + half_size[2]]
                    
                    # Create cube wireframe
                    wireframe_edges = [
                        ((0,0,0), (1,0,0)),
                        ((0,0,0), (0,1,0)),
                        ((0,0,0), (0,0,1)),
                        ((1,1,0), (0,1,0)),
                        ((1,1,0), (1,0,0)),
                        ((1,1,0), (1,1,1)),
                        ((0,1,1), (0,1,0)),
                        ((0,1,1), (0,0,1)),
                        ((0,1,1), (1,1,1)),
                        ((1,0,1), (1,0,0)),
                        ((1,0,1), (0,0,1)),
                        ((1,0,1), (1,1,1))
                    ]
                    
                    for (xs, ys, zs), (xe, ye, ze) in wireframe_edges:
                        self.ax.plot3D(
                            [x[xs], x[xe]], 
                            [y[ys], y[ye]], 
                            [z[zs], z[ze]], 
                            color='gray', 
                            alpha=0.5
                        )
                
                elif obstacle_type == 'cylinder':
                    # Cylinder plotting logic (if needed)
                    pass
            
            except Exception as e:
                print(f"Could not plot obstacle: {e}")
    
    def visualize_paths(self, 
                        paths: List[List[Tuple[float, float, float]]], 
                        start_positions: List[Tuple[float, float, float]],
                        goal_positions: List[Tuple[float, float, float]],
                        algorithm_name: Optional[str] = None,
                        save_path: Optional[str] = None):
        """
        Visualize static paths for multiple agents.
        
        Args:
            paths: List of paths for each agent
            start_positions: Starting positions of agents
            goal_positions: Goal positions of agents
            algorithm_name: Name of the algorithm used for pathfinding
            save_path: Optional path to save the visualization
        """
        # Clear previous plot
        self.ax.clear()
        
        # Set environment bounds
        self.ax.set_xlim(0, self.environment.size[0])
        self.ax.set_ylim(0, self.environment.size[1])
        self.ax.set_zlim(0, self.environment.size[2])
        
        # Label axes
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Plot obstacles
        self._plot_obstacles()
        
        # Color palette for agents
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        
        # Plot paths and markers
        for i, (path, start, goal) in enumerate(zip(paths, start_positions, goal_positions)):
            color = colors[i % len(colors)]
            
            # Convert path to numpy array for easier manipulation
            path_array = np.array(path)
            
            # Plot path
            self.ax.plot3D(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                           color=color, linewidth=2, label=f'Agent {i+1}')
            
            # Plot start and goal points
            self.ax.scatter(*start, color=color, marker='o', s=100, edgecolor='black')
            self.ax.scatter(*goal, color=color, marker='x', s=100, edgecolor='black')
        
        self.ax.legend()
        
        # Update title with algorithm name
        title = f'Paths in {self.environment.name}'
        if algorithm_name:
            title += f' ({algorithm_name})'
        plt.title(title)
        
        plt.tight_layout()
        
        # If save_path is provided, save the animation
        if save_path:
            self.save_animation(anim, algorithm_name or 'unknown', save_path)
        
        plt.show()
        
        # Return the animation object for potential later use
        return anim
    
    def animate_paths(self, 
                      paths: List[List[Tuple[float, float, float]]], 
                      start_positions: List[Tuple[float, float, float]],
                      goal_positions: List[Tuple[float, float, float]],
                      algorithm_name: Optional[str] = None,
                      save_path: Optional[str] = None):
        """
        Animate paths for multiple agents.
        
        Args:
            paths: List of paths for each agent
            start_positions: Starting positions of agents
            goal_positions: Goal positions of agents
            algorithm_name: Name of the algorithm used for pathfinding
            save_path: Optional path to save the animation
        """
        # Determine max path length for animation
        max_path_length = max(len(path) for path in paths)
        
        # Clear previous plot
        self.ax.clear()
        
        # Set environment bounds
        self.ax.set_xlim(0, self.environment.size[0])
        self.ax.set_ylim(0, self.environment.size[1])
        self.ax.set_zlim(0, self.environment.size[2])
        
        # Label axes
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Plot obstacles
        self._plot_obstacles()
        
        # Color palette for agents
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        
        # Plot fixed start and goal markers
        for i, (start, goal) in enumerate(zip(start_positions, goal_positions)):
            color = colors[i % len(colors)]
            
            # Plot start point (green circle)
            self.ax.scatter(
                start[0], start[1], start[2], 
                color='green', 
                marker='o', 
                s=100, 
                edgecolor='black',
                label='Start' if i == 0 else None
            )
            
            # Plot goal point (red cross)
            self.ax.scatter(
                goal[0], goal[1], goal[2], 
                color='red', 
                marker='x', 
                s=100, 
                edgecolor='black',
                label='Goal' if i == 0 else None
            )
        
        # Prepare path lines for animation
        path_lines = []
        for i, path in enumerate(paths):
            color = colors[i % len(colors)]
            
            # Prepare path line
            line, = self.ax.plot3D([], [], [], color=color, linewidth=2, label=f'Agent {i+1} Path')
            path_lines.append(line)
        
        # Add legend
        self.ax.legend()
        
        # Update title with algorithm name
        title = f'Animated Paths in {self.environment.name}'
        if algorithm_name:
            title += f' ({algorithm_name})'
        plt.title(title)
        
        def init():
            """Initialize animation."""
            for line in path_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return path_lines
        
        def animate(frame):
            """Animate paths."""
            for i, (path, line) in enumerate(zip(paths, path_lines)):
                # Ensure path is long enough
                if frame < len(path):
                    # Update path line
                    line_data = path[:frame+1]
                    line_array = np.array(line_data)
                    line.set_data(line_array[:, 0], line_array[:, 1])
                    line.set_3d_properties(line_array[:, 2])
            
            return path_lines
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, 
            animate, 
            init_func=init,
            frames=max_path_length, 
            interval=200,  # milliseconds between frames
            blit=True
        )
        
        plt.tight_layout()
        
        # If save_path is provided, save the animation
        if save_path:
            self.save_animation(anim, algorithm_name or 'unknown', save_path)
        
        plt.show()
        
        # Return the animation object for potential later use
        return anim
    
    def save_animation(self, anim, algorithm_name: str, save_path: str):
        """
        Save animation from different views (front, top, side).
        """
        # Create the save directory if it doesn't exist
        import os
        os.makedirs(save_path, exist_ok=True)
        
        views = {
            'front': (0, 0),
            'top': (0, 90),
            'side': (90, 0)
        }
        
        for view_name, (elev, azim) in views.items():
            self.ax.view_init(elev=elev, azim=azim)
            anim.save(f"{save_path}/{algorithm_name}_{view_name}.mp4", writer='ffmpeg')
            print(f"Saved animation: {save_path}/{algorithm_name}_{view_name}.mp4")

