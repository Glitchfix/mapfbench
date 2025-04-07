# Environment Configurations

This directory contains JSON configuration files for different multiagent pathfinding environments.

## File Format

Each JSON file represents a complete environment configuration with the following possible keys:

- `name`: A descriptive name for the environment
- `size`: Dimensions of the environment (x, y, z)
- `constraints`: Environment movement and boundary constraints
  - `max_velocity`: Maximum movement speed
  - `collision_radius`: Collision detection radius
  - `boundary_behavior`: How agents interact with environment boundaries
- `obstacles`: List of obstacle configurations
  - `type`: Shape of the obstacle (sphere, cube, cylinder)
  - `center`: Central position of the obstacle
  - Shape-specific parameters:
    - Sphere: `radius`
    - Cube: `size`
    - Cylinder: `radius`, `height`, `axis`
- `dynamic_obstacles`: Optional list of moving obstacles

## Example Environment Configuration

```json
{
    "name": "Complex Obstacle Environment",
    "size": [100, 100, 100],
    "constraints": {
        "max_velocity": 2,
        "collision_radius": 1.5,
        "boundary_behavior": "reflect"
    },
    "obstacles": [
        {
            "type": "sphere",
            "center": [25, 25, 25],
            "radius": 5
        },
        {
            "type": "cube", 
            "center": [50, 50, 50],
            "size": [10, 10, 10]
        }
    ]
}
```

## Example Obstacles

### Sphere Obstacle
```json
{
  "type": "sphere",
  "center": [25, 25, 25],
  "radius": 5
}
```

### Cube Obstacle
```json
{
  "type": "cube", 
  "center": [50, 50, 50],
  "size": [10, 10, 10]
}
```

### Cylinder Obstacle
```json
{
  "type": "cylinder",
  "center": [75, 75, 75],
  "radius": 3,
  "height": 15,
  "axis": "y"
}
```

## Usage

Load an environment configuration in your code:

```python
env = Environment.load_from_json('path/to/environment.json')
print(env.name)  # Access the environment name
```

## Naming Conventions

- Use descriptive names that reflect the environment's characteristics
- Names can include information about obstacle types, terrain, or purpose
- Maximum length recommended: 50 characters
