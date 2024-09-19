import taichi as ti

@ti.data_oriented
class Grid:
    def __init__(self, grid_size, cell_size):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.node_count_x = int(grid_size[0] / cell_size)
        self.node_count_y = int(grid_size[1] / cell_size)
        self.node_count = (self.node_count_x, self.node_count_y)
        self.total_nodes = self.node_count_x * self.node_count_y

        # Initialize Taichi fields
        self.position = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.mass = ti.field(dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.mass_body1 = ti.field(dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.mass_body2 = ti.field(dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.velocity_body1 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.velocity_body2 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.momentum = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.momentum_body1 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.momentum_body2 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.force = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.force_body1 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.force_body2 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.acceleration = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.acceleration_body1 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.acceleration_body2 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.normals_body1 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))
        self.normals_body2 = ti.Vector.field(2, dtype=ti.f32, shape=(self.node_count_x, self.node_count_y))

        # Initialize body_ids as a flat array representing a 2D grid
        self.body_id = ti.field(dtype=ti.i32, shape=(self.total_nodes*2))
        self.body_id.fill(-1)
        self.initialize_grid()

    @ti.kernel
    def initialize_grid(self):
        for i, j in ti.ndrange(self.node_count_x, self.node_count_y):
            x = i * self.cell_size
            y = j * self.cell_size
            self.position[i, j] = ti.Vector([x, y])

    @ti.kernel
    def reset_nodes(self):
        for i, j in ti.ndrange(self.node_count_x, self.node_count_y):
            self.mass[i, j] = 0.0
            self.mass_body1[i, j] = 0.0
            self.mass_body2[i, j] = 0.0
            self.velocity[i, j] = ti.Vector([0.0, 0.0])
            self.velocity_body1[i, j] = ti.Vector([0.0, 0.0])
            self.velocity_body2[i, j] = ti.Vector([0.0, 0.0])
            self.momentum[i, j] = ti.Vector([0.0, 0.0])
            self.momentum_body1[i, j] = ti.Vector([0.0, 0.0])
            self.momentum_body2[i, j] = ti.Vector([0.0, 0.0])
            self.force[i, j] = ti.Vector([0.0, 0.0])
            self.force_body1[i, j] = ti.Vector([0.0, 0.0])
            self.force_body2[i, j] = ti.Vector([0.0, 0.0])
            self.acceleration[i, j] = ti.Vector([0.0, 0.0])
            self.acceleration_body1[i, j] = ti.Vector([0.0, 0.0])
            self.acceleration_body2[i, j] = ti.Vector([0.0, 0.0])
            self.normals_body1[i, j] = ti.Vector([0.0, 0.0])
            self.normals_body2[i, j] = ti.Vector([0.0, 0.0])
            self.body_id[i * self.node_count_y + j] = -1  # Reset to -1
            self.body_id[i * self.node_count_y + j + self.total_nodes] = -1  # Reset to -1
    @ti.func
    def get_node(self, i, j):
        # Returns a tuple of node properties at (i, j)
        position = self.position[i, j]
        mass = self.mass[i, j]
        mass_body1 = self.mass_body1[i, j]
        mass_body2 = self.mass_body2[i, j]
        velocity = self.velocity[i, j]
        velocity_body1 = self.velocity_body1[i, j]
        velocity_body2 = self.velocity_body2[i, j]
        momentum = self.momentum[i, j]
        momentum_body1 = self.momentum_body1[i, j]
        momentum_body2 = self.momentum_body2[i, j]
        force = self.force[i, j]
        acceleration = self.acceleration[i, j]
        acceleration_body1 = self.acceleration_body1[i, j]
        acceleration_body2 = self.acceleration_body2[i, j]
        normals_body1 = self.normals_body1[i, j]
        normals_body2 = self.normals_body2[i, j]
        node_index = i * self.node_count_y + j
        body_ids = ti.Vector([self.body_id[node_index], self.body_id[node_index + self.total_nodes]])
        return (position, mass, mass_body1, mass_body2, velocity, velocity_body1, velocity_body2,
                momentum, momentum_body1, momentum_body2, force, acceleration, acceleration_body1,
                acceleration_body2, normals_body1, normals_body2, body_ids)

    @ti.func
    def set_node(self, i, j, node):
        (position, mass, mass_body1, mass_body2, velocity, velocity_body1, velocity_body2,
         momentum, momentum_body1, momentum_body2, force, acceleration, acceleration_body1,
         acceleration_body2, normals_body1, normals_body2, body_ids) = node

        self.position[i, j] = position
        self.mass[i, j] = mass
        self.mass_body1[i, j] = mass_body1
        self.mass_body2[i, j] = mass_body2
        self.velocity[i, j] = velocity
        self.velocity_body1[i, j] = velocity_body1
        self.velocity_body2[i, j] = velocity_body2
        self.momentum[i, j] = momentum
        self.momentum_body1[i, j] = momentum_body1
        self.momentum_body2[i, j] = momentum_body2
        self.force[i, j] = force
        self.acceleration[i, j] = acceleration
        self.acceleration_body1[i, j] = acceleration_body1
        self.acceleration_body2[i, j] = acceleration_body2
        self.normals_body1[i, j] = normals_body1
        self.normals_body2[i, j] = normals_body2
        node_index = i * self.node_count_y + j
        self.body_id[node_index] = body_ids[0]
        self.body_id[node_index + self.total_nodes] = body_ids[1]

    @ti.func
    def get_nearby_nodes(self, particle_position):
        i = int(particle_position[0] / self.cell_size)
        j = int(particle_position[1] / self.cell_size)
        nearby_nodes = ti.Matrix.zero(ti.i32, 25, 2)  # 5x5 neighborhood
        idx = 0
        for di in ti.static(range(-2, 3)):
            for dj in ti.static(range(-2, 3)):
                ni = i + di
                nj = j + dj
                if 0 <= ni < self.node_count_x and 0 <= nj < self.node_count_y:
                    nearby_nodes[idx, 0] = ni
                    nearby_nodes[idx, 1] = nj
                else:
                    nearby_nodes[idx, 0] = -1
                    nearby_nodes[idx, 1] = -1
                idx += 1
        return nearby_nodes

    @ti.func
    def get_surrounding_nodes(self, particle_position, psep):
        index_x = int((particle_position[0] - (psep / 2.0) - self.position[0, 0][0]) / self.cell_size)
        index_y = int((particle_position[1] - (psep / 2.0) - self.position[0, 0][1]) / self.cell_size)
        surrounding_nodes = ti.Matrix.zero(ti.i32, 9, 2)  # 3x3 neighborhood
        idx = 0
        for j in ti.static(range(3)):
            for i in ti.static(range(3)):
                ni = index_x + i
                nj = index_y + j
                if 0 <= ni < self.node_count_x and 0 <= nj < self.node_count_y:
                    surrounding_nodes[idx, 0] = ni
                    surrounding_nodes[idx, 1] = nj
                else:
                    surrounding_nodes[idx, 0] = -1
                    surrounding_nodes[idx, 1] = -1  # Default to (-1, -1) if out of bounds
                idx += 1
        return surrounding_nodes

    # Additional methods for updating grid nodes can be added here
