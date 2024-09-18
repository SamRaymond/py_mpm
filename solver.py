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

    def step(self):
        self.prepare_step()
        self.particle_to_grid()
        self.update_grid()
        self.grid_to_particle()
        self.update_particles()
        self.time += self.dt

    def prepare_step(self):
        self.grid.reset_nodes()
        self.particles.reset_particles()

    def particle_to_grid(self):
        for p_idx in range(self.particles.num_particles):
            particle_pos = self.particles.position[p_idx]
            i = int(particle_pos[0] / self.grid.cell_size)
            j = int(particle_pos[1] / self.grid.cell_size)

            for di in range(-2, 3):
                for dj in range(-2, 3):
                    cell_i = i + di
                    cell_j = j + dj

                    # Ensure cell_i and cell_j are within valid bounds
                    if 0 <= cell_i < self.grid.node_count[0] and 0 <= cell_j < self.grid.node_count[1]:
                        cell_ID = cell_i + cell_j * self.grid.node_count[0]

                        # Debug statements to trace the values
                        # if cell_ID >= self.grid.total_nodes or cell_ID < 0:
                        #     print(f"Debug: cell_i={cell_i}, cell_j={cell_j}, cell_ID={cell_ID}, total_nodes={self.grid.total_nodes}")
                        #     continue

                        node_pos = self.grid.position[cell_i, cell_j]

                        shape_x, shape_y = shape_function(particle_pos, node_pos, self.grid.cell_size)
                        grad_shape_x, grad_shape_y = gradient_shape_function(particle_pos, node_pos, self.grid.cell_size)

                        shape_value = shape_x * shape_y
                        mp_mass = self.particles.mass[p_idx]
                        mp_density = self.particles.density[p_idx]
                        mp_obj_id = self.particles.object_id[p_idx]
                        # print(f"Debug: mp_obj_id={mp_obj_id}, self.grid.body_id[2*cell_ID]={self.grid.body_id[2*cell_ID]}, self.grid.body_id[2*cell_ID + 1]={self.grid.body_id[2*cell_ID + 1]}")

                        # Contact/Multibody handling
                        if self.grid.body_id[2*cell_ID] == -1:
                            self.grid.body_id[2*cell_ID] = mp_obj_id
                        elif self.grid.body_id[2*cell_ID + 1] == -1 and self.grid.body_id[2*cell_ID] != mp_obj_id:
                            self.grid.body_id[2*cell_ID + 1] = mp_obj_id

                        # Update node properties for center of mass
                        self.grid.mass[cell_i, cell_j] += mp_mass * shape_value
                        self.grid.momentum[cell_i, cell_j] += mp_mass * self.particles.velocity[p_idx] * shape_value

                        force_x = -(mp_mass / mp_density) * (
                            grad_shape_x * shape_y * self.particles.stress[p_idx, 0, 0] + 
                            grad_shape_y * shape_x * self.particles.stress[p_idx, 0, 1]
                        )
                        force_y = -(mp_mass / mp_density) * (
                            grad_shape_x * shape_y * self.particles.stress[p_idx, 1, 0] + 
                            grad_shape_y * shape_x * self.particles.stress[p_idx, 1, 1]
                        )

                        self.grid.force[cell_i, cell_j][0] += force_x
                        self.grid.force[cell_i, cell_j][1] += force_y

                        # Update body1 properties
                        if self.grid.body_id[2*cell_ID] == mp_obj_id:
                            # print(f"Debug: mp_obj_id={mp_obj_id}, self.grid.body_id[2*cell_ID]={self.grid.body_id[2*cell_ID]}")
                            self.grid.mass_body1[cell_i, cell_j] += mp_mass * shape_value
                            self.grid.momentum_body1[cell_i, cell_j] += mp_mass * self.particles.velocity[p_idx] * shape_value
                            self.grid.force_body1[cell_i, cell_j][0] += force_x
                            self.grid.force_body1[cell_i, cell_j][1] += force_y
                            self.grid.normals_body1[cell_i, cell_j][0] += grad_shape_x * shape_y * self.particles.volume[p_idx]
                            self.grid.normals_body1[cell_i, cell_j][1] += shape_x * grad_shape_y * self.particles.volume[p_idx]

                        # Update body2 properties
                        elif self.grid.body_id[2*cell_ID + 1] == mp_obj_id:
                            # print(f"Debug: mp_obj_id={mp_obj_id}, self.grid.body_id[2*cell_ID + 1]={self.grid.body_id[2*cell_ID + 1]}")
                            self.grid.mass_body2[cell_i, cell_j] += mp_mass * shape_value
                            self.grid.momentum_body2[cell_i, cell_j] += mp_mass * self.particles.velocity[p_idx] * shape_value
                            self.grid.force_body2[cell_i, cell_j][0] += force_x
                            self.grid.force_body2[cell_i, cell_j][1] += force_y
                            self.grid.normals_body2[cell_i, cell_j][0] += grad_shape_x * shape_y * self.particles.volume[p_idx]
                            self.grid.normals_body2[cell_i, cell_j][1] += shape_x * grad_shape_y * self.particles.volume[p_idx]

    def update_grid(self):
        dt = self.dt / 2 if self.time <= self.dt else self.dt
        gravity = np.array([0, 0])  # Gravity is off for now
        for i in range(self.grid.node_count[0]):
            for j in range(self.grid.node_count[1]):
                cell_ID = i + j * self.grid.node_count[0]
                
                if self.grid.mass[i, j] > 1e-9:
                    # Calculate center of mass velocity
                    self.grid.velocity[i, j] = self.grid.momentum[i, j] / self.grid.mass[i, j]

                    # Normalize normal vectors
                    mag1 = np.sqrt(self.grid.normals_body1[i, j, 0]**2 + self.grid.normals_body1[i, j, 1]**2)
                    mag2 = np.sqrt(self.grid.normals_body2[i, j, 0]**2 + self.grid.normals_body2[i, j, 1]**2)

                    if mag1 > 1e-8:
                        self.grid.normals_body1[i, j] /= mag1
                    else:
                        self.grid.normals_body1[i, j] = [0.0, 0.0]

                    if mag2 > 1e-8:
                        self.grid.normals_body2[i, j] /= mag2
                    else:
                        self.grid.normals_body2[i, j] = [0.0, 0.0]

                    # Contact Detection
                    if self.grid.body_id[2*cell_ID+1] != -1:  # only perform when two bodies are detected at a node
                        approach1 = approach2 = 0.0
                        rel_vel_body1 = [0.0, 0.0]
                        rel_vel_body2 = [0.0, 0.0]


                        if self.grid.mass_body1[i, j] != 0.0:
                            rel_vel_body1 = [
                                self.grid.momentum_body1[i, j, 0] / self.grid.mass_body1[i, j] - self.grid.velocity[i, j, 0],
                                self.grid.momentum_body1[i, j, 1] / self.grid.mass_body1[i, j] - self.grid.velocity[i, j, 1]
                            ]
                            approach1 = (rel_vel_body1[0] * self.grid.normals_body1[i, j, 0] +
                                        rel_vel_body1[1] * self.grid.normals_body1[i, j, 1])

                        if self.grid.mass_body2[i, j] != 0.0:
                            rel_vel_body2 = [
                                self.grid.momentum_body2[i, j, 0] / self.grid.mass_body2[i, j] - self.grid.velocity[i, j, 0],
                                self.grid.momentum_body2[i, j, 1] / self.grid.mass_body2[i, j] - self.grid.velocity[i, j, 1]
                            ]
                            approach2 = (rel_vel_body2[0] * self.grid.normals_body2[i, j, 0] +
                                        rel_vel_body2[1] * self.grid.normals_body2[i, j, 1])

                        is_approach = (approach1 > 0.01) or (approach2 > 0.01)

                        if is_approach:  # two bodies are moving toward each other
                            # print(f"Debug: approach1={approach1}, approach2={approach2}")
                            # Update body1
                            if self.grid.mass_body1[i, j] != 0.0:
                                # Placeholder for Coulomb friction calculation
                                contact_normals = self.grid.normals_body1[i, j]
                                # contact_normals = self.grid.normals_body1[i, j]
                                friction_normals = [0.0, 0.0]
                                mu_prime = 0.0
                                # Replace the above with actual implementation

                                self.grid.velocity_body1[i, j] = [
                                    self.grid.momentum_body1[i, j, 0] / self.grid.mass_body1[i, j] - approach1 * contact_normals[0],
                                    self.grid.momentum_body1[i, j, 1] / self.grid.mass_body1[i, j] - approach1 * contact_normals[1]
                                ]
                                self.grid.acceleration_body1[i, j] = [
                                    self.grid.force_body1[i, j, 0] / self.grid.mass_body1[i, j] - (approach1 * contact_normals[0]) / dt,
                                    self.grid.force_body1[i, j, 1] / self.grid.mass_body1[i, j] - (approach1 * contact_normals[1]) / dt
                                ]

                                if self.time <= self.dt:
                                    self.grid.velocity_body1[i, j, 0] += self.grid.acceleration_body1[i, j, 0] * 0.5 * dt
                                    self.grid.velocity_body1[i, j, 1] += self.grid.acceleration_body1[i, j, 1] * 0.5 * dt
                                else:
                                    self.grid.velocity_body1[i, j, 0] += self.grid.acceleration_body1[i, j, 0] * dt
                                    self.grid.velocity_body1[i, j, 1] += self.grid.acceleration_body1[i, j, 1] * dt

                            # Update body2
                            if self.grid.mass_body2[i, j] != 0.0:
                                # Placeholder for Coulomb friction calculation
                                contact_normals = self.grid.normals_body2[i, j]
                                # contact_normals = self.grid.normals_body2[i, j]
                                friction_normals = [0.0, 0.0]
                                mu_prime = 0.0
                                # Replace the above with actual implementation

                                self.grid.velocity_body2[i, j] = [
                                    self.grid.momentum_body2[i, j, 0] / self.grid.mass_body2[i, j] - approach2 * contact_normals[0],
                                    self.grid.momentum_body2[i, j, 1] / self.grid.mass_body2[i, j] - approach2 * contact_normals[1]
                                ]
                                self.grid.acceleration_body2[i, j] = [
                                    self.grid.force_body2[i, j, 0] / self.grid.mass_body2[i, j] - (approach2 * contact_normals[0]) / dt,
                                    self.grid.force_body2[i, j, 1] / self.grid.mass_body2[i, j] - (approach2 * contact_normals[1]) / dt
                                ]

                                if self.time <= self.dt:
                                    self.grid.velocity_body2[i, j, 0] += self.grid.acceleration_body2[i, j, 0] * 0.5 * dt
                                    self.grid.velocity_body2[i, j, 1] += self.grid.acceleration_body2[i, j, 1] * 0.5 * dt
                                else:
                                    self.grid.velocity_body2[i, j, 0] += self.grid.acceleration_body2[i, j, 0] * dt
                                    self.grid.velocity_body2[i, j, 1] += self.grid.acceleration_body2[i, j, 1] * dt

                        else:  # Bodies aren't approaching, so they can be updated separately
                            # Update body1
                            if self.grid.mass_body1[i, j] != 0.0:
                                self.grid.velocity_body1[i, j] = [
                                    self.grid.momentum_body1[i, j, 0] / self.grid.mass_body1[i, j],
                                    self.grid.momentum_body1[i, j, 1] / self.grid.mass_body1[i, j]
                                ]
                                self.grid.acceleration_body1[i, j] = [
                                    self.grid.force_body1[i, j, 0] / self.grid.mass_body1[i, j],
                                    self.grid.force_body1[i, j, 1] / self.grid.mass_body1[i, j]
                                ]

                                if self.time <= self.dt:
                                    self.grid.velocity_body1[i, j, 0] += self.grid.acceleration_body1[i, j, 0] * 0.5 * dt
                                    self.grid.velocity_body1[i, j, 1] += self.grid.acceleration_body1[i, j, 1] * 0.5 * dt
                                else:
                                    self.grid.velocity_body1[i, j, 0] += self.grid.acceleration_body1[i, j, 0] * dt
                                    self.grid.velocity_body1[i, j, 1] += self.grid.acceleration_body1[i, j, 1] * dt

                            # Update body2
                            if self.grid.mass_body2[i, j] != 0.0:
                                self.grid.velocity_body2[i, j] = [
                                    self.grid.momentum_body2[i, j, 0] / self.grid.mass_body2[i, j],
                                    self.grid.momentum_body2[i, j, 1] / self.grid.mass_body2[i, j]
                                ]
                                self.grid.acceleration_body2[i, j] = [
                                    self.grid.force_body2[i, j, 0] / self.grid.mass_body2[i, j],
                                    self.grid.force_body2[i, j, 1] / self.grid.mass_body2[i, j]
                                ]

                                if self.time <= self.dt:
                                    self.grid.velocity_body2[i, j, 0] += self.grid.acceleration_body2[i, j, 0] * 0.5 * dt
                                    self.grid.velocity_body2[i, j, 1] += self.grid.acceleration_body2[i, j, 1] * 0.5 * dt
                                else:
                                    self.grid.velocity_body2[i, j, 0] += self.grid.acceleration_body2[i, j, 0] * dt
                                    self.grid.velocity_body2[i, j, 1] += self.grid.acceleration_body2[i, j, 1] * dt

                    else:  # only one body detected at this node
                        if self.grid.mass_body1[i, j] != 0.0:
                            self.grid.velocity_body1[i, j] = [
                                self.grid.momentum_body1[i, j, 0] / self.grid.mass_body1[i, j],
                                self.grid.momentum_body1[i, j, 1] / self.grid.mass_body1[i, j]
                            ]
                            self.grid.acceleration_body1[i, j] = [
                                self.grid.force_body1[i, j, 0] / self.grid.mass_body1[i, j],
                                self.grid.force_body1[i, j, 1] / self.grid.mass_body1[i, j]
                            ]

                            if self.time <= self.dt:
                                self.grid.velocity_body1[i, j, 0] += self.grid.acceleration_body1[i, j, 0] * 0.5 * dt
                                self.grid.velocity_body1[i, j, 1] += self.grid.acceleration_body1[i, j, 1] * 0.5 * dt
                            else:
                                self.grid.velocity_body1[i, j, 0] += self.grid.acceleration_body1[i, j, 0] * dt
                                self.grid.velocity_body1[i, j, 1] += self.grid.acceleration_body1[i, j, 1] * dt

                    # Update center of mass fields
                    self.grid.force[i, j] += self.grid.mass[i, j] * gravity
                    self.grid.momentum[i, j] += self.grid.force[i, j] * dt
                    self.grid.velocity[i, j] = self.grid.momentum[i, j] / self.grid.mass[i, j]

    def grid_to_particle(self):
        for p_idx in range(self.particles.num_particles):
            particle_velocity_update = np.zeros(2)
            strain_rate = np.zeros((2, 2))
            acceleration_update = np.zeros(2)
            density_rate = 0
            normal_update = np.zeros(2)
            velocity = np.zeros(2)
            acceleration = np.zeros(2)
            normals = np.zeros(2)
            particle_pos = self.particles.position[p_idx]
            mp_obj_id = self.particles.object_id[p_idx]
            nearby_nodes = self.grid.get_nearby_nodes(particle_pos)
            
            for node in nearby_nodes:
                if node['mass'] > 0:
                    shape_x, shape_y = shape_function(particle_pos, node['position'], self.grid.cell_size)
                    grad_shape_x, grad_shape_y = gradient_shape_function(particle_pos, node['position'], self.grid.cell_size)

                    shape_value = shape_x * shape_y
                    
                    if node['body_id'][0] == mp_obj_id:
                        velocity[0] = node['velocity_body1'][0]
                        velocity[1] = node['velocity_body1'][1]
                        acceleration[0] = node['acceleration_body1'][0]
                        acceleration[1] = node['acceleration_body1'][1]
                        normals[0] = node['normals_body1'][0]
                        normals[1] = node['normals_body1'][1]
                    elif node['body_id'][1] == mp_obj_id:
                        velocity[0] = node['velocity_body2'][0]
                        velocity[1] = node['velocity_body2'][1]
                        acceleration[0] = node['acceleration_body2'][0]
                        acceleration[1] = node['acceleration_body2'][1]
                        normals[0] = node['normals_body2'][0]
                        normals[1] = node['normals_body2'][1]
                    else:
                        continue  # Skip if particle doesn't belong to either body at this node

                    density_rate -= self.particles.density[p_idx] * (
                        (grad_shape_x * shape_y * velocity[0]) + 
                        (grad_shape_y * shape_x * velocity[1])
                    )

                    acceleration_update[0] += shape_value * acceleration[0]
                    acceleration_update[1] += shape_value * acceleration[1]
                    particle_velocity_update[0] += shape_value * velocity[0]
                    particle_velocity_update[1] += shape_value * velocity[1]

                    # Calculate strain rate components
                    strain_rate[0, 0] += grad_shape_x * shape_y * velocity[0]  # exx
                    strain_rate[0, 1] += 0.5 * (grad_shape_x * shape_y * velocity[1] + grad_shape_y * shape_x * velocity[0])  # exy
                    strain_rate[1, 0] = strain_rate[0, 1]  # eyx = exy
                    strain_rate[1, 1] += grad_shape_y * shape_x * velocity[1]  # eyy

                    # Update normals
                    normal_update += shape_value * normals[:2]  # Only use x and y components

            # Update particle properties
            self.particles.Gvelocity[p_idx] = particle_velocity_update
            self.particles.acceleration[p_idx] = acceleration_update
            self.particles.strain_rate[p_idx] = strain_rate
            self.particles.density_rate[p_idx] = density_rate
            # self.particles.normals[p_idx] = normal_update

    def update_particles(self):
        for p_idx in range(self.particles.num_particles):
            # Update velocity using both grid-interpolated velocity and acceleration
            self.particles.velocity[p_idx] += self.particles.acceleration[p_idx] * self.dt
            # Update position
            self.particles.position[p_idx] += self.particles.Gvelocity[p_idx] * self.dt
            self.particles.density[p_idx] += self.particles.density_rate[p_idx] * self.dt
            self.particles.volume[p_idx] = self.particles.mass[p_idx] / self.particles.density[p_idx]

            # Update stress using the material model
            material = self.particles.materials[p_idx]
            stress_rate = material.compute_stress_rate(self.particles.strain_rate[p_idx])
            self.particles.stress[p_idx] += stress_rate * self.dt

    def apply_boundary_conditions(self):
        pass
        # Implement boundary conditions here