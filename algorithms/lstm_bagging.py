import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional

from mapfbench.algorithms.base_algorithm import BaseAlgorithm
from mapfbench.core.environment import Environment
from mapfbench.core.agent import Agent

class LSTMPathModel(nn.Module):
    """
    LSTM Neural Network for Path Prediction
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Size of input features (3D coordinates)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state
        
        Returns:
            Output prediction and new hidden state
        """
        # LSTM forward pass
        out, hidden = self.lstm(x, hidden)
        
        # Predict next coordinate
        out = self.fc(out[:, -1, :])
        
        return out, hidden

class LSTMBaggingPathfinder(BaseAlgorithm):
    """
    LSTM Bagging Pathfinding Algorithm.
    
    Uses an ensemble of LSTM networks to predict optimal paths.
    """
    
    def __init__(self, 
                 environment: Optional[Environment] = None, 
                 agents: Optional[List[Agent]] = None,
                 num_models: int = 5,
                 lstm_units: int = 64,
                 epochs: int = 50):
        """
        Initialize LSTM Bagging Pathfinder.
        
        Args:
            environment: Environment for pathfinding (optional)
            agents: List of agents to find paths for (optional)
            num_models: Number of LSTM models in the ensemble
            lstm_units: Number of LSTM units in each model
            epochs: Training epochs for each model
        """
        super().__init__(environment, agents)
        
        # Determine device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.num_models = num_models
        self.lstm_units = lstm_units
        self.epochs = epochs
        
        self.models = []
        self.collision_checks = 0
    
    def _normalize_path(self, path):
        """
        Normalize path coordinates to [0, 1] range.
        
        Args:
            path: List of path coordinates
        
        Returns:
            Normalized path
        """
        normalized = []
        for point in path:
            normalized_point = [
                point[i] / self.environment.size[i] 
                for i in range(3)
            ]
            normalized.append(normalized_point)
        return normalized
    
    def _prepare_training_data(self, paths):
        """
        Prepare training data for LSTM models.
        
        Args:
            paths: List of existing paths in the environment
        
        Returns:
            Torch tensors for training
        """
        # Normalize paths
        normalized_paths = [self._normalize_path(path) for path in paths]
        
        # Pad sequences
        max_len = max(len(path) for path in normalized_paths)
        padded_paths = []
        
        for path in normalized_paths:
            # Pad or truncate
            if len(path) < max_len:
                padding = [[0, 0, 0]] * (max_len - len(path))
                padded_path = path + padding
            else:
                padded_path = path[:max_len]
            padded_paths.append(padded_path)
        
        # Convert to torch tensor
        X = torch.tensor(padded_paths, dtype=torch.float32).to(self.device)
        
        # Prepare input and target sequences
        X_input = X[:, :-1, :]
        y_target = X[:, 1:, :]
        
        return X_input, y_target
    
    def _train_ensemble(self, paths):
        """
        Train an ensemble of LSTM models.
        
        Args:
            paths: List of existing paths to train on
        """
        # Prepare training data
        X, y = self._prepare_training_data(paths)
        
        # Create and train models
        self.models = []
        for _ in range(self.num_models):
            # Initialize model
            model = LSTMPathModel(
                input_size=3, 
                hidden_size=self.lstm_units
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters())
            
            # Training loop
            for _ in range(self.epochs):
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = model(X)
                loss = criterion(outputs, y[:, -1, :])
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            self.models.append(model)
    
    def _predict_path(self, start, goal):
        """
        Predict path using ensemble of LSTM models.
        
        Args:
            start: Starting position
            goal: Goal position
        
        Returns:
            Predicted path
        """
        # Normalize start and goal
        norm_start = torch.tensor(
            [start[i] / self.environment.size[i] for i in range(3)], 
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(self.device)
        
        norm_goal = [goal[i] / self.environment.size[i] for i in range(3)]
        
        # Predict using ensemble
        predictions = []
        for model in self.models:
            # Predict path
            path = [start]
            current_input = norm_start
            hidden = None
            
            for _ in range(50):  # Max path length
                # Collision check
                self.collision_checks += 1
                
                # Predict next position
                with torch.no_grad():
                    next_pos_norm, hidden = model(current_input, hidden)
                
                # Denormalize
                next_pos = tuple(
                    next_pos_norm.cpu().numpy()[0][i] * self.environment.size[i] 
                    for i in range(3)
                )
                
                # Check if goal reached
                if np.linalg.norm(
                    np.array(next_pos) - np.array(goal)
                ) < self.environment.constraints.get('collision_radius', 1.0):
                    path.append(goal)
                    break
                
                # Check if position is valid
                if not self.environment.is_valid_position(next_pos):
                    break
                
                path.append(next_pos)
                
                # Update input for next iteration
                current_input = torch.tensor(
                    [next_pos_norm.cpu().numpy()[0]], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
            
            predictions.append(path)
        
        # Ensemble voting (average of predictions)
        max_len = max(len(p) for p in predictions)
        ensemble_path = []
        
        for step in range(max_len):
            step_positions = [
                p[step] if step < len(p) else p[-1] 
                for p in predictions
            ]
            
            # Average position
            avg_pos = tuple(
                np.mean([pos[i] for pos in step_positions]) 
                for i in range(3)
            )
            
            ensemble_path.append(avg_pos)
        
        return ensemble_path
    
    def solve(self):
        """
        Solve paths for all agents using LSTM Bagging.
        """
        # If no pre-trained models, train on initial paths
        if not self.models:
            initial_paths = [
                self._generate_initial_path(agent.start, agent.goal) 
                for agent in self.agents
            ]
            self._train_ensemble(initial_paths)
        
        # Predict paths for agents
        for agent in self.agents:
            path = self._predict_path(agent.start, agent.goal)
            agent.path = path
            agent.update_position(path[-1] if path else agent.start)
    
    def _generate_initial_path(self, start, goal):
        """
        Generate an initial path using a simple heuristic.
        
        Args:
            start: Starting position
            goal: Goal position
        
        Returns:
            Initial path
        """
        path = [start]
        current = start
        
        while np.linalg.norm(
            np.array(current) - np.array(goal)
        ) > self.environment.constraints.get('collision_radius', 1.0):
            # Move towards goal
            direction = np.array(goal) - np.array(current)
            step = direction / np.linalg.norm(direction)
            
            new_pos = tuple(current[i] + step[i] for i in range(3))
            
            # Check if new position is valid
            if not self.environment.is_valid_position(new_pos):
                # If blocked, try alternative moves
                neighbors = self.environment.get_neighbors(current, goal)
                if not neighbors:
                    break
                new_pos = neighbors[0]
            
            path.append(new_pos)
            current = new_pos
        
        path.append(goal)
        return path
