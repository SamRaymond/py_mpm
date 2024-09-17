import numpy as np
import matplotlib.pyplot as plt

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
    L = cell_size
    lp = cell_size / 4

    def piecewise_shape(xp, xv):
        x_diff = xp - xv
        if x_diff <= -L - lp:
            return 0
        elif -L - lp < x_diff <= -L + lp:
            return ((L + lp + x_diff) ** 2) / (4 * L * lp)
        elif -L + lp < x_diff <= -lp:
            return 1 + (x_diff / L)
        elif -lp < x_diff <= lp:
            return 1 - ((x_diff ** 2 + lp ** 2) / (2 * L * lp))
        elif lp < x_diff <= L - lp:
            return 1 - (x_diff / L)
        elif L - lp < x_diff <= L + lp:
            return ((L + lp - x_diff) ** 2) / (4 * L * lp)
        else:
            return 0

    sx = piecewise_shape(particle_pos[0], node_pos[0])
    sy = piecewise_shape(particle_pos[1], node_pos[1])

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
    L = cell_size
    lp = cell_size / 4

    def piecewise_gradient(xp, xv):
        x_diff = xp - xv
        if x_diff <= -L - lp:
            return 0
        elif -L - lp < x_diff <= -L + lp:
            return (L + lp + x_diff) / (2 * L * lp)
        elif -L + lp < x_diff <= -lp:
            return 1 / L
        elif -lp < x_diff <= lp:
            return -x_diff / (L * lp)
        elif lp < x_diff <= L - lp:
            return -1 / L
        elif L - lp < x_diff <= L + lp:
            return -(L + lp - x_diff) / (2 * L * lp)
        else:
            return 0

    dsx_dx = piecewise_gradient(particle_pos[0], node_pos[0])
    dsy_dy = piecewise_gradient(particle_pos[1], node_pos[1])

    return dsx_dx, dsy_dy
# wrap in main
if __name__ == "__main__":  
    # Parameters
    cell_size = 1.0
    node_pos = np.array([0.0, 0.0])
    particle_positions = np.linspace(-1.5 * cell_size, 1.5 * cell_size, 500)

    # Calculate shape function and gradient values
    shape_values = [shape_function(np.array([x, 0.0]), node_pos, cell_size)[0] for x in particle_positions]
    gradient_values = [gradient_shape_function(np.array([x, 0.0]), node_pos, cell_size)[0] for x in particle_positions]

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(particle_positions, shape_values, label='Shape Function')
    plt.title('Shape Function')
    plt.xlabel('Particle Position')
    plt.ylabel('Shape Function Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(particle_positions, gradient_values, label='Gradient of Shape Function', color='r')
    plt.title('Gradient of Shape Function')
    plt.xlabel('Particle Position')
    plt.ylabel('Gradient Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

