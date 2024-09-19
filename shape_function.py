import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, default_fp=ti.f32)

# Constants
cell_size = 1.0
L = cell_size
lp = cell_size / 4.0
node_pos = ti.Vector([0.0, 0.0])
num_particles = 500

# Taichi fields
particle_positions = ti.field(dtype=ti.f32, shape=num_particles)
shape_values = ti.field(dtype=ti.f32, shape=num_particles)
gradient_values = ti.field(dtype=ti.f32, shape=num_particles)

# Initialize particle positions with dtype=np.float32
particle_positions_np = np.linspace(-1.5 * cell_size, 1.5 * cell_size, num_particles, dtype=np.float32)
particle_positions.from_numpy(particle_positions_np)

@ti.func
def piecewise_shape(x_diff, cell_size):
    L = cell_size
    lp = cell_size / 4.0
    # x_diff = -x_diff 
    result = 0.0  # Initialize result
    if x_diff <= -L - lp:
        result = 0
    elif -L - lp < x_diff <= -L + lp:
        result = ((L + lp + x_diff) ** 2) / (4 * L * lp)
    elif -L + lp < x_diff <= -lp:
        result = 1 + (x_diff / L)
    elif -lp < x_diff <= lp:
        result = 1 - ((x_diff ** 2 + lp ** 2) / (2 * L * lp))
    elif lp < x_diff <= L - lp:
        result = 1 - (x_diff / L)
    elif L - lp < x_diff <= L + lp:
        result = ((L + lp - x_diff) ** 2) / (4 * L * lp)
    else:
        result = 0
    # if x_diff <= -L - lp:
    #     result = 0.0
    # elif x_diff <= -L + lp:
    #     result = ((L + lp + x_diff) ** 2) / (4 * L * lp)
    # elif x_diff <= -lp:
    #     result = 1.0 + (x_diff / L)
    # elif x_diff <= lp:
    #     result = 1.0 - ((x_diff ** 2 + lp ** 2) / (2 * L * lp))
    # elif x_diff <= L - lp:
    #     result = 1.0 - (x_diff / L)
    # elif x_diff <= L + lp:
    #     result = ((L + lp - x_diff) ** 2) / (4 * L * lp)
    # else:
    #     result = 0.0
    return result

@ti.func
def piecewise_gradient(x_diff, cell_size):
    L = cell_size
    lp = cell_size / 4.0
    # x_diff = -x_diff 
    result = 0.0  # Initialize result
    if x_diff <= -L - lp:
        result =  0
    elif -L - lp < x_diff <= -L + lp:
        result =  (L + lp + x_diff) / (2 * L * lp)
    elif -L + lp < x_diff <= -lp:
        result =  1 / L
    elif -lp < x_diff <= lp:
        result =  -x_diff / (L * lp)
    elif lp < x_diff <= L - lp:
        result =  -1 / L
    elif L - lp < x_diff <= L + lp:
        result =  -(L + lp - x_diff) / (2 * L * lp)
    else:
        result =  0

    # if x_diff <= -L - lp:
    #     result = 0.0
    # elif x_diff <= -L + lp:
    #     result = (L + lp + x_diff) / (2 * L * lp)
    # elif x_diff <= -lp:
    #     result = 1.0 / L
    # elif x_diff <= lp:
    #     result = -x_diff / (L * lp)
    # elif x_diff <= L - lp:
    #     result = -1.0 / L
    # elif x_diff <= L + lp:
    #     result = -(L + lp - x_diff) / (2 * L * lp)
    # else:
    #     result = 0.0
    return result

@ti.func
def shape_function(particle_pos, node_pos, cell_size):
    x_diff = particle_pos[0] - node_pos[0]
    y_diff = particle_pos[1] - node_pos[1]
    shape_x = piecewise_shape(x_diff, cell_size)
    shape_y = piecewise_shape(y_diff, cell_size)
    return shape_x, shape_y

@ti.func
def gradient_shape_function(particle_pos, node_pos, cell_size):
    x_diff = particle_pos[0] - node_pos[0]
    y_diff = particle_pos[1] - node_pos[1]
    grad_shape_x = piecewise_gradient(x_diff, cell_size)
    grad_shape_y = piecewise_gradient(y_diff, cell_size)
    return grad_shape_x, grad_shape_y

# If you're not using the following code for plotting or testing, you can comment it out
# @ti.kernel
# def compute_shape_and_gradient():
#     for i in particle_positions:
#         xp = particle_positions[i]
#         x_diff = xp - node_pos[0]
#         y_diff = 0.0 - node_pos[1]  # Particle y-position is constant at 0.0

#         # Compute shape function values
#         sx = piecewise_shape(x_diff, cell_size)
#         sy = piecewise_shape(y_diff, cell_size)
#         shape_values[i] = sx  # Store sx (since sy is constant)

#         # Compute gradient values
#         dsx_dx = piecewise_gradient(x_diff, cell_size)
#         gradient_values[i] = dsx_dx  # Store dsx_dx

# # Main execution
# if __name__ == "__main__":
#     compute_shape_and_gradient()
#     shape_values_np = shape_values.to_numpy()
#     gradient_values_np = gradient_values.to_numpy()
#     # Plotting code...

