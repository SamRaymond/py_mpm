import numpy as np

def shape_function(particle_pos, node_pos, cell_size):
    """
    Calculate the linear nodal shape function value (tent function).
    
    Args:
    particle_pos (np.array): Position of the particle (2D)
    node_pos (np.array): Position of the grid node (2D)
    cell_size (float): Size of a grid cell
    
    Returns:
    tuple: Shape function values (sx, sy)
    """
    dx = (particle_pos[0] - node_pos[0]) / cell_size
    dy = (particle_pos[1] - node_pos[1]) / cell_size
    
    sx = 1 - abs(dx) if abs(dx) < 1 else 0
    sy = 1 - abs(dy) if abs(dy) < 1 else 0
    
    return sx, sy

def gradient_shape_function(particle_pos, node_pos, cell_size):
    """
    Calculate the gradient of the linear nodal shape function.
    
    Args:
    particle_pos (np.array): Position of the particle (2D)
    node_pos (np.array): Position of the grid node (2D)
    cell_size (float): Size of a grid cell
    
    Returns:
    tuple: Gradient of the shape function (dsx_dx, dsy_dy)
    """
    dx = (particle_pos[0] - node_pos[0]) / cell_size
    dy = (particle_pos[1] - node_pos[1]) / cell_size
    
    dsx_dx = -1 if -1 < dx < 0 else (1 if 0 <= dx < 1 else 0)
    dsy_dy = -1 if -1 < dy < 0 else (1 if 0 <= dy < 1 else 0)
    
    return dsx_dx / cell_size, dsy_dy / cell_size

