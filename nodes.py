import numpy as np

class Grid:
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.node_count = (int(grid_size[0] / cell_size), int(grid_size[1] / cell_size))
        self.total_nodes = self.node_count[0] * self.node_count[1]
        
        # Initialize body_id with the correct size
        self.body_id = np.full((2 * self.total_nodes,), -1, dtype=int)
        
        self.physical_size = (grid_size[0], grid_size[1])
        self.position = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.mass = np.zeros((self.node_count[0], self.node_count[1]))
        self.mass_body1 = np.zeros((self.node_count[0], self.node_count[1]))
        self.mass_body2 = np.zeros((self.node_count[0], self.node_count[1]))
        self.velocity = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.velocity_body1 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.velocity_body2 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.momentum = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.momentum_body1 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.momentum_body2 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.force = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.force_body1 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.force_body2 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.acceleration_body1 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.acceleration_body2 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.acceleration = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.normals_body1 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.normals_body2 = np.zeros((self.node_count[0], self.node_count[1], 2))
        self.initialize_grid()

    def initialize_grid(self):
        for i in range(self.node_count[0]):
            for j in range(self.node_count[1]):
                x = i * self.cell_size
                y = j * self.cell_size
                self.position[i, j] = [x, y]

    def reset_nodes(self):
        self.mass.fill(0)
        self.mass_body1.fill(0)
        self.mass_body2.fill(0)
        self.velocity.fill(0)
        self.velocity_body1.fill(0)
        self.velocity_body2.fill(0)
        self.body_id.fill(-1)  # Reset to -1 instead of 0
        self.momentum.fill(0)
        self.momentum_body1.fill(0)
        self.momentum_body2.fill(0)
        self.force.fill(0)
        self.force_body1.fill(0)
        self.force_body2.fill(0)
        self.acceleration.fill(0)
        self.acceleration_body1.fill(0)
        self.acceleration_body2.fill(0)
        self.normals_body1.fill(0)
        self.normals_body2.fill(0)

    def get_node(self, i, j):
        cell_ID = i + j * self.node_count[0]
        return {
            'position': self.position[i, j],
            'mass': self.mass[i, j],
            'mass_body1': self.mass_body1[i, j],
            'mass_body2': self.mass_body2[i, j],
            'velocity': self.velocity[i, j],
            'velocity_body1': self.velocity_body1[i, j],
            'velocity_body2': self.velocity_body2[i, j],
            'momentum': self.momentum[i, j],
            'momentum_body1': self.momentum_body1[i, j],
            'momentum_body2': self.momentum_body2[i, j],
            'force': self.force[i, j],
            'acceleration': self.acceleration[i, j],
            'acceleration_body1': self.acceleration_body1[i, j],
            'acceleration_body2': self.acceleration_body2[i, j],
            'normals_body1': self.normals_body1[i, j],
            'normals_body2': self.normals_body2[i, j],
            'body_id': self.body_id[2*cell_ID:2*cell_ID+2]  # Access the correct slice of the 1D array
        }

    def set_node(self, i, j, node):
        cell_ID = i + j * self.node_count[0]
        self.body_id[2 * cell_ID] = node['body_id'][0]
        self.body_id[2 * cell_ID + 1] = node['body_id'][1]
        self.mass[i, j] = node['mass']
        self.mass_body1[i, j] = node['mass_body1']
        self.mass_body2[i, j] = node['mass_body2']
        self.velocity[i, j] = node['velocity']
        self.velocity_body1[i, j] = node['velocity_body1']
        self.velocity_body2[i, j] = node['velocity_body2']
        self.momentum[i, j] = node['momentum']
        self.momentum_body1[i, j] = node['momentum_body1']
        self.momentum_body2[i, j] = node['momentum_body2']
        self.force[i, j] = node['force']
        self.force_body1[i, j] = node['force_body1']
        self.force_body2[i, j] = node['force_body2']
        self.acceleration[i, j] = node['acceleration']
        self.acceleration_body1[i, j] = node['acceleration_body1']
        self.acceleration_body2[i, j] = node['acceleration_body2']
        self.normals_body1[i, j] = node['normals_body1']
        self.normals_body2[i, j] = node['normals_body2']

    def get_nearby_nodes(self, particle_position):
        i = int(particle_position[0] / self.cell_size)
        j = int(particle_position[1] / self.cell_size)
        nearby_nodes = []
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if 0 <= i + di < self.node_count[0] and 0 <= j + dj < self.node_count[1]:
                    nearby_nodes.append(self.get_node(i + di, j + dj))
        return nearby_nodes

    def get_surrounding_nodes(self, particle_position, psep):
        index_x = int((particle_position[0] - (psep / 2) - self.position[0, 0, 0]) / self.cell_size)
        index_y = int((particle_position[1] - (psep / 2) - self.position[0, 0, 1]) / self.cell_size)
        surrounding_nodes = []
        for j in range(3):
            for i in range(3):
                ni = index_x + i
                nj = index_y + j
                if 0 <= ni < self.node_count[0] and 0 <= nj < self.node_count[1]:
                    surrounding_nodes.append(self.get_node(ni, nj))
                else:
                    surrounding_nodes.append(self.get_node(0, 0))  # Default to (0, 0) if out of bounds
        return surrounding_nodes

class Node:
    def __init__(self, position):
        self.position = position
        self.mass = 0.0
        self.mass_body1 = 0.0
        self.mass_body2 = 0.0
        self.velocity = np.zeros(2)
        self.velocity_body1 = np.zeros(2)
        self.velocity_body2 = np.zeros(2)
        self.momentum = np.zeros(2)
        self.momentum_body1 = np.zeros(2)
        self.momentum_body2 = np.zeros(2)
        self.force = np.zeros(2)
        self.force_body1 = np.zeros(2)
        self.force_body2 = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.acceleration_body1 = np.zeros(2)
        self.acceleration_body2 = np.zeros(2)
        self.normals_body1 = np.zeros(2)
        self.normals_body2 = np.zeros(2)
        self.body_id = np.full((2,), -1, dtype=int)

    def reset(self):
        self.mass = 0.0
        self.mass_body1 = 0.0
        self.mass_body2 = 0.0
        self.velocity.fill(0)
        self.velocity_body1.fill(0)
        self.velocity_body2.fill(0)
        self.momentum.fill(0)
        self.momentum_body1.fill(0)
        self.momentum_body2.fill(0)
        self.force.fill(0)
        self.force_body1.fill(0)
        self.force_body2.fill(0)
        self.acceleration.fill(0)
        self.acceleration_body1.fill(0)
        self.acceleration_body2.fill(0)
        self.normals_body1.fill(0)
        self.normals_body2.fill(0)
        self.body_id.fill(-1)
        # Reset other properties as needed