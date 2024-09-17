import numpy as np
from particles import Particles
from nodes import Grid
from shape_function import shape_function, gradient_shape_function


class MPMSolver:
    def __init__(self, particles: Particles, grid: Grid, dt: float):
        self.particles = particles
        self.grid = grid
        self.dt = dt
        self.time = 0.0
        
        # Initialize initial_volume and initial_density for each particle
        for particle in self.particles.particles:
            if 'initial_volume' not in particle:
                particle['initial_volume'] = particle['volume']
            if 'initial_density' not in particle:
                particle['initial_density'] = particle['density']

    def step(self):
        self.prepare_step()
        self.particle_to_grid()
        self.update_grid()
        self.grid_to_particle()
        self.update_particles()
        # self.apply_boundary_conditions()
        self.time += self.dt

    def prepare_step(self):
        self.grid.reset_nodes()
        
        # Initialize node_particle_map as a 2D list of lists
        self.grid.node_particle_map = [[[] for _ in range(self.grid.node_count[1])] 
                                       for _ in range(self.grid.node_count[0])]

        for p_idx in range(len(self.particles.particles)):
            particle_pos = self.particles.particles[p_idx]['position']
            i = int(particle_pos[0] / self.grid.cell_size)
            j = int(particle_pos[1] / self.grid.cell_size)

            node_range = 2  # Influence radius

            for di in range(-node_range, node_range + 1):
                for dj in range(-node_range, node_range + 1):
                    node_i = i + di
                    node_j = j + dj

                    if 0 <= node_i < self.grid.node_count[0] and 0 <= node_j < self.grid.node_count[1]:
                        node_pos = np.array([node_i * self.grid.cell_size, node_j * self.grid.cell_size])
                        shape_x, shape_y = shape_function(particle_pos, node_pos, self.grid.cell_size)

                        if shape_x * shape_y > 0:
                            self.grid.node_particle_map[node_i][node_j].append(p_idx)

    def particle_to_grid(self):
        for i in range(self.grid.node_count[0]):
            for j in range(self.grid.node_count[1]):
                node = self.grid.nodes[i, j]
                node_pos = np.array([i * self.grid.cell_size, j * self.grid.cell_size])

                node.force = np.zeros(2)
                node.mass = 0
                node.momentum = np.zeros(2)

                particle_indices = self.grid.node_particle_map[i][j]

                for p_idx in particle_indices:
                    particle = self.particles.get_particle(p_idx)
                    particle_pos = particle['position']

                    shape_x, shape_y = shape_function(particle_pos, node_pos, self.grid.cell_size)
                    grad_shape_x, grad_shape_y = gradient_shape_function(particle_pos, node_pos, self.grid.cell_size)

                    shape_value = shape_x * shape_y

                    # Update node properties
                    node.mass += particle['mass'] * shape_value
                    node.momentum += particle['mass'] * particle['velocity'] * shape_value

                    force_x = -(particle['mass'] / particle['density']) * (
                        grad_shape_x * shape_y * particle['stress'][0, 0] + 
                        grad_shape_y * shape_x * particle['stress'][0, 1]
                    )
                    force_y = -(particle['mass'] / particle['density']) * (
                        grad_shape_x * shape_y * particle['stress'][1, 0] + 
                        grad_shape_y * shape_x * particle['stress'][1, 1]
                    )

                    node.force[0] += force_x
                    node.force[1] += force_y

    def update_grid(self):
        dt = self.dt / 2 if self.time <= self.dt else self.dt
        gravity = np.array([0, 0])  # Gravity is off for now
        for node in self.grid.nodes.flat:
            if node.mass > 1e-9:
                node.force += node.mass * gravity
                node.momentum += node.force * dt
                node.velocity = node.momentum / node.mass

        # Clear forces for the next step
        # for node in self.grid.nodes.flat:
        #     node.force = np.zeros(2)

    def grid_to_particle(self):
      for p_idx, particle in enumerate(self.particles.particles):
            particle_velocity_update = np.zeros(2)
            strain_rate = np.zeros((2, 2))
            acceleration_update = np.zeros(2)
            nearby_nodes = self.grid.get_nearby_nodes(particle['position'])
            density_rate = 0
            for node in nearby_nodes:
                if node.mass > 1e-9:
                    shape_x, shape_y = shape_function(particle['position'], node.position, self.grid.cell_size)
                    grad_shape_x, grad_shape_y = gradient_shape_function(particle['position'], node.position, self.grid.cell_size)

                    shape_value = shape_x * shape_y
                    particle_velocity_update += shape_value * node.velocity
                    acceleration_update += shape_value * node.force / node.mass
                    density_rate -= particle['density']*((grad_shape_x*shape_y*node.velocity[0])+(grad_shape_y*shape_x*node.velocity[1]))
                    # Calculate strain rate components
                    strain_rate[0, 0] += grad_shape_x * shape_y * node.velocity[0]  # exx
                    strain_rate[0, 1] += 0.5 * (grad_shape_x * shape_y * node.velocity[1] + grad_shape_y * shape_x * node.velocity[0])  # exy
                    strain_rate[1, 0] = strain_rate[0, 1]  # eyx = exy
                    strain_rate[1, 1] += grad_shape_y * shape_x * node.velocity[1]  # eyy

            # Update particle velocity
            particle['Gvelocity'] = particle_velocity_update
            # Update acceleration
            particle['acceleration'] = acceleration_update
            # Store strain rate in particle
            particle['strain_rate'] = strain_rate
            # Store density rate in particle
            particle['density_rate'] = density_rate
    def update_particles(self):

        for particle in self.particles.particles:
            # Update velocity using both grid-interpolated velocity and acceleration
            particle['velocity'] += particle['acceleration'] * self.dt
            # Update position
            particle['position'] += particle['velocity'] * self.dt

            # particle['density'] += particle['density_rate'] * self.dt
            # particle['volume'] = particle['mass'] / particle['density']
            # Update deformation gradient (F)
            # identity = np.eye(2)
            # particle['F'] = np.dot(identity + self.dt * particle['strain_rate'], particle.get('F', identity))

            # # Update volume
            # J = np.linalg.det(particle['F'])
            # particle['volume'] = particle['initial_volume'] * J

            # Update density
            # particle['density'] = particle['initial_density'] / J

            # Update stress using the material model
            material = particle['material']
            stress_rate = material.compute_stress_rate(particle['strain_rate'])
            particle['stress'] += stress_rate * self.dt

    def apply_boundary_conditions(self):
        pass
        # Implement boundary conditions here

