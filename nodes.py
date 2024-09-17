import numpy as np

class Grid:
    def __init__(self, physical_size, cell_size):
        self.physical_size = np.array(physical_size)
        self.cell_size = cell_size
        self.node_count = np.ceil(self.physical_size / self.cell_size).astype(int) + 1
        self.nodes = self.initialize_grid()
        self.total_nodes = np.prod(self.node_count)

    def initialize_grid(self):
        grid = np.empty(self.node_count, dtype=object)
        for i in range(self.node_count[0]):
            for j in range(self.node_count[1]):
                x = i * self.cell_size
                y = j * self.cell_size
                grid[i, j] = Node(position=np.array([x, y]))
        return grid

    def reset_nodes(self):
        for i in range(self.node_count[0]):
            for j in range(self.node_count[1]):
                self.nodes[i, j].reset()

    def get_node(self, i, j):
        return self.nodes[i, j]

    def set_node(self, i, j, node):
        self.nodes[i, j] = node
    
    def get_nearby_nodes(self, particle_position):
        i = int(particle_position[0] / self.cell_size)
        j = int(particle_position[1] / self.cell_size)
        nearby_nodes = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if 0 <= i + di < self.node_count[0] and 0 <= j + dj < self.node_count[1]:
                    nearby_nodes.append(self.nodes[i + di, j + dj])
        return nearby_nodes

    @property
    def node_count_total(self):
        return self.total_nodes

class Node:
    def __init__(self, position):
        self.position = position
        self.mass = 0.0
        self.velocity = np.zeros(2)
        self.momentum = np.zeros(2)
        self.force = np.zeros(2)

    def reset(self):
        self.mass = 0.0
        self.momentum.fill(0)
        self.velocity.fill(0)
        self.force.fill(0)
        # Reset other properties as needed