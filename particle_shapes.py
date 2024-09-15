import numpy as np

class ParticleShape:
    def __init__(self, cell_size, object_id):
        self.cell_size = cell_size
        self.particles_per_cell = 4  # 4 particles per cell in 2D
        self.object_id = object_id

    def generate_particles(self):
        raise NotImplementedError("Subclasses must implement generate_particles method")

    def _generate_grid_particles(self, x_range, y_range):
        particles = []
        for i in range(x_range[0], x_range[1]):
            for j in range(y_range[0], y_range[1]):
                for px in range(2):
                    for py in range(2):
                        # Place particles at 1/4 and 3/4 of the cell in each dimension
                        x = (i + (px + 1) / 3) * self.cell_size
                        y = (j + (py + 1) / 3) * self.cell_size
                        
                        if self._is_inside(x, y):
                            particle = {
                                'position': np.array([x, y]),
                                'velocity': np.zeros(2),
                                'mass': self.density * self.cell_size**2 / self.particles_per_cell,
                                'volume': self.cell_size**2 / self.particles_per_cell,
                                'stress': np.zeros((2, 2)),
                                'strain': np.zeros((2, 2)),
                                'object_id': self.object_id,  # Add object_id to the particle
                            }
                            particles.append(particle)
        return particles

    def _is_inside(self, x, y):
        raise NotImplementedError("Subclasses must implement _is_inside method")

class Block(ParticleShape):
    def __init__(self, cell_size, width, height, position, object_id):
        super().__init__(cell_size, object_id)
        self.width = width
        self.height = height
        self.position = np.array(position)

    def generate_particles(self):
        num_cells_x = int(self.width / self.cell_size)
        num_cells_y = int(self.height / self.cell_size)
        
        x_range = (0, num_cells_x)
        y_range = (0, num_cells_y)
        
        return self._generate_grid_particles(x_range, y_range)

    def _is_inside(self, x, y):
        return (0 <= x < self.width) and (0 <= y < self.height)

class Disk(ParticleShape):
    def __init__(self, cell_size, radius, center, object_id):
        super().__init__(cell_size, object_id)
        self.radius = radius
        self.center = np.array(center)
        print(f"Disk created with radius={radius}, center={center}")

    def generate_particles(self):
        particles = []
        x_min, x_max = self.center[0] - self.radius, self.center[0] + self.radius
        y_min, y_max = self.center[1] - self.radius, self.center[1] + self.radius
        
        for i in range(int(x_min / self.cell_size), int(x_max / self.cell_size) + 1):
            for j in range(int(y_min / self.cell_size), int(y_max / self.cell_size) + 1):
                for px in range(2):
                    for py in range(2):
                        x = (i + (px + 1) / 3) * self.cell_size
                        y = (j + (py + 1) / 3) * self.cell_size
                        if self._is_inside(x, y):
                            particle = {
                                'position': np.array([x, y]),
                                'velocity': np.zeros(2),
                                'mass': 1.0 ,
                                'volume': 1.0,
                                'stress': np.zeros((2, 2)),
                                'strain': np.zeros((2, 2)),
                                'object_id': self.object_id,  # Add object_id to the particle
                            }
                            particles.append(particle)
        
        print(f"Disk generated {len(particles)} particles")
        if len(particles) == 0:
            print(f"Warning: No particles generated. Check disk parameters.")
        return particles

    def _is_inside(self, x, y):
        dx = x - self.center[0]
        dy = y - self.center[1]
        return dx**2 + dy**2 <= self.radius**2


