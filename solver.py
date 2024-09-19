import taichi as ti
from particles import Particles
from nodes import Grid
from shape_function import shape_function, gradient_shape_function

@ti.data_oriented
class MPMSolver:
    def __init__(self, particles: Particles, grid: Grid, dt: float):
        self.particles = particles
        self.grid = grid
        self.dt = dt
        self.time = 0.0
        self.total_nodes = self.grid.node_count_x * self.grid.node_count_y

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


    @ti.kernel
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
                    if 0 <= cell_i < self.grid.node_count_x and 0 <= cell_j < self.grid.node_count_y:
                        node_pos = self.grid.position[cell_i, cell_j]

                        shape_x, shape_y = shape_function(particle_pos, node_pos, self.grid.cell_size)
                        grad_shape_x, grad_shape_y = gradient_shape_function(particle_pos, node_pos, self.grid.cell_size)

                        shape_value = shape_x * shape_y
                        mp_mass = self.particles.mass[p_idx]
                        mp_density = self.particles.density[p_idx]
                        mp_obj_id = self.particles.object_id[p_idx]

                        # Contact/Multibody handling
                        if self.grid.body_id[cell_i * self.grid.node_count_y + cell_j] == -1:
                            self.grid.body_id[cell_i * self.grid.node_count_y + cell_j] = mp_obj_id
                        elif self.grid.body_id[cell_i * self.grid.node_count_y + cell_j + self.grid.total_nodes] == -1 and self.grid.body_id[cell_i * self.grid.node_count_y + cell_j] != mp_obj_id:
                            self.grid.body_id[cell_i * self.grid.node_count_y + cell_j + self.grid.total_nodes] = mp_obj_id

                        # Update node properties for center of mass
                        ti.atomic_add(self.grid.mass[cell_i, cell_j], mp_mass * shape_value)
                        ti.atomic_add(self.grid.momentum[cell_i, cell_j][0], mp_mass * self.particles.velocity[p_idx][0] * shape_value)
                        ti.atomic_add(self.grid.momentum[cell_i, cell_j][1], mp_mass * self.particles.velocity[p_idx][1] * shape_value)

                        force_x = -(mp_mass / mp_density) * (
                            grad_shape_x * shape_y * self.particles.stress[p_idx][0, 0] +
                            grad_shape_y * shape_x * self.particles.stress[p_idx][0, 1]
                        )
                        force_y = -(mp_mass / mp_density) * (
                            grad_shape_x * shape_y * self.particles.stress[p_idx][1, 0] +
                            grad_shape_y * shape_x * self.particles.stress[p_idx][1, 1]
                        )
                        # force = ti.Vector([force_x, force_y])
                        ti.atomic_add(self.grid.force[cell_i, cell_j][0], force_x)
                        ti.atomic_add(self.grid.force[cell_i, cell_j][1], force_y)

                        volume_weight = self.particles.volume[p_idx] * ti.Vector([grad_shape_x * shape_y, shape_x * grad_shape_y])

                        # Update body1 properties
                        if self.grid.body_id[cell_i * self.grid.node_count_y + cell_j] == mp_obj_id:
                            ti.atomic_add(self.grid.mass_body1[cell_i, cell_j], mp_mass * shape_value)
                            ti.atomic_add(self.grid.momentum_body1[cell_i, cell_j], mp_mass * self.particles.velocity[p_idx] * shape_value)
                            ti.atomic_add(self.grid.force_body1[cell_i, cell_j][0], force_x)
                            ti.atomic_add(self.grid.force_body1[cell_i, cell_j][1], force_y)
                            ti.atomic_add(self.grid.normals_body1[cell_i, cell_j], volume_weight)

                        # Update body2 properties
                        elif self.grid.body_id[cell_i * self.grid.node_count_y + cell_j + self.grid.total_nodes] == mp_obj_id:
                            ti.atomic_add(self.grid.mass_body2[cell_i, cell_j], mp_mass * shape_value)
                            ti.atomic_add(self.grid.momentum_body2[cell_i, cell_j][0], mp_mass * self.particles.velocity[p_idx][0] * shape_value)
                            ti.atomic_add(self.grid.momentum_body2[cell_i, cell_j][1], mp_mass * self.particles.velocity[p_idx][1] * shape_value)
                            ti.atomic_add(self.grid.force_body2[cell_i, cell_j][0], force_x)
                            ti.atomic_add(self.grid.force_body2[cell_i, cell_j][1], force_y)
                            ti.atomic_add(self.grid.normals_body2[cell_i, cell_j], volume_weight)

    @ti.kernel
    def update_grid(self):
        dt = self.dt / 2 if self.time <= self.dt else self.dt
        gravity = ti.Vector([0.0, 0.0])  # Gravity is off for now

        for i, j in ti.ndrange(self.grid.node_count_x, self.grid.node_count_y):
            if self.grid.mass[i, j] > 0:
                # Calculate center of mass velocity
                self.grid.velocity[i, j] = self.grid.momentum[i, j] / self.grid.mass[i, j]

                # Normalize normal vectors
                mag1 = self.grid.normals_body1[i, j].norm()
                mag2 = self.grid.normals_body2[i, j].norm()

                if mag1 > 1e-8:
                    self.grid.normals_body1[i, j] /= mag1
                else:
                    self.grid.normals_body1[i, j] = ti.Vector([0.0, 0.0])

                if mag2 > 1e-8:
                    self.grid.normals_body2[i, j] /= mag2
                else:
                    self.grid.normals_body2[i, j] = ti.Vector([0.0, 0.0])

                # Contact Detection
                if self.grid.body_id[i * self.grid.node_count_y + j +1] != -1:  # Two bodies detected at node
                    approach1 = 0.0
                    approach2 = 0.0
                    rel_vel_body1 = ti.Vector([0.0, 0.0])
                    rel_vel_body2 = ti.Vector([0.0, 0.0])

                    if self.grid.mass_body1[i, j] != 0.0:
                        rel_vel_body1 = (self.grid.momentum_body1[i, j] / self.grid.mass_body1[i, j]) - self.grid.velocity[i, j]
                        approach1 = rel_vel_body1.dot(self.grid.normals_body1[i, j])

                    if self.grid.mass_body2[i, j] != 0.0:
                        rel_vel_body2 = (self.grid.momentum_body2[i, j] / self.grid.mass_body2[i, j]) - self.grid.velocity[i, j]
                        approach2 = rel_vel_body2.dot(self.grid.normals_body2[i, j])

                    is_approach = (approach1 > 0.01) or (approach2 > 0.01)

                    if is_approach:
                        # Update body1
                        if self.grid.mass_body1[i, j] > 0.0:
                            contact_normals = self.grid.normals_body1[i, j]
                            approach = approach1
                            self.grid.velocity_body1[i, j][0] = (self.grid.momentum_body1[i, j][0] / self.grid.mass_body1[i, j]) - approach * contact_normals[0]
                            self.grid.velocity_body1[i, j][1] = (self.grid.momentum_body1[i, j][1] / self.grid.mass_body1[i, j]) - approach * contact_normals[1]
                            self.grid.acceleration_body1[i, j][0] = (self.grid.force_body1[i, j][0] / self.grid.mass_body1[i, j]) - (approach * contact_normals[0]) / dt
                            self.grid.acceleration_body1[i, j][1] = (self.grid.force_body1[i, j][1] / self.grid.mass_body1[i, j]) - (approach * contact_normals[1]) / dt

                            if self.time <= self.dt:
                                self.grid.velocity_body1[i, j][0] += self.grid.acceleration_body1[i, j][0] * 0.5 * dt
                                self.grid.velocity_body1[i, j][1] += self.grid.acceleration_body1[i, j][1] * 0.5 * dt
                            else:
                                self.grid.velocity_body1[i, j][0] += self.grid.acceleration_body1[i, j][0] * dt
                                self.grid.velocity_body1[i, j][1] += self.grid.acceleration_body1[i, j][1] * dt

                        # Update body2
                        if self.grid.mass_body2[i, j] != 0.0:
                            contact_normals = self.grid.normals_body2[i, j]
                            approach = approach2
                            self.grid.velocity_body2[i, j][0] = (self.grid.momentum_body2[i, j][0] / self.grid.mass_body2[i, j]) - approach * contact_normals[0]
                            self.grid.velocity_body2[i, j][1] = (self.grid.momentum_body2[i, j][1] / self.grid.mass_body2[i, j]) - approach * contact_normals[1]
                            self.grid.acceleration_body2[i, j][0] = (self.grid.force_body2[i, j][0] / self.grid.mass_body2[i, j]) - (approach * contact_normals[0]) / dt
                            self.grid.acceleration_body2[i, j][1] = (self.grid.force_body2[i, j][1] / self.grid.mass_body2[i, j]) - (approach * contact_normals[1]) / dt

                            if self.time <= self.dt:
                                self.grid.velocity_body2[i, j][0] += self.grid.acceleration_body2[i, j][0] * 0.5 * dt
                                self.grid.velocity_body2[i, j][1] += self.grid.acceleration_body2[i, j][1] * 0.5 * dt
                            else:
                                self.grid.velocity_body2[i, j][0] += self.grid.acceleration_body2[i, j][0] * dt
                                self.grid.velocity_body2[i, j][1] += self.grid.acceleration_body2[i, j][1] * dt

                    else:
                        # Update body1
                        if self.grid.mass_body1[i, j] > 0.0:
                            self.grid.velocity_body1[i, j][0] = self.grid.momentum_body1[i, j][0] / self.grid.mass_body1[i, j]
                            self.grid.velocity_body1[i, j][1] = self.grid.momentum_body1[i, j][1] / self.grid.mass_body1[i, j]
                            self.grid.acceleration_body1[i, j][0] = self.grid.force_body1[i, j][0] / self.grid.mass_body1[i, j]
                            self.grid.acceleration_body1[i, j][1] = self.grid.force_body1[i, j][1] / self.grid.mass_body1[i, j]

                            if self.time <= self.dt:
                                self.grid.velocity_body1[i, j][0] += self.grid.acceleration_body1[i, j][0] * 0.5 * dt
                                self.grid.velocity_body1[i, j][1] += self.grid.acceleration_body1[i, j][1] * 0.5 * dt
                            else:
                                self.grid.velocity_body1[i, j][0] += self.grid.acceleration_body1[i, j][0] * dt
                                self.grid.velocity_body1[i, j][1] += self.grid.acceleration_body1[i, j][1] * dt

                        # Update body2
                        if self.grid.mass_body2[i, j] >0.0:
                            self.grid.velocity_body2[i, j][0] = self.grid.momentum_body2[i, j][0] / self.grid.mass_body2[i, j]
                            self.grid.velocity_body2[i, j][1] = self.grid.momentum_body2[i, j][1] / self.grid.mass_body2[i, j]
                            self.grid.acceleration_body2[i, j][0] = self.grid.force_body2[i, j][0] / self.grid.mass_body2[i, j]
                            self.grid.acceleration_body2[i, j][1] = self.grid.force_body2[i, j][1] / self.grid.mass_body2[i, j]

                            if self.time <= self.dt:
                                self.grid.velocity_body2[i, j][0] += self.grid.acceleration_body2[i, j][0] * 0.5 * dt
                                self.grid.velocity_body2[i, j][1] += self.grid.acceleration_body2[i, j][1] * 0.5 * dt
                            else:
                                self.grid.velocity_body2[i, j][0] += self.grid.acceleration_body2[i, j][0] * dt
                                self.grid.velocity_body2[i, j][1] += self.grid.acceleration_body2[i, j][1] * dt

                else:
                    # Only one body at this node
                    if self.grid.mass_body1[i, j] > 0.0:
                        self.grid.velocity_body1[i, j][0] = self.grid.momentum_body1[i, j][0] / self.grid.mass_body1[i, j]
                        self.grid.velocity_body1[i, j][1] = self.grid.momentum_body1[i, j][1] / self.grid.mass_body1[i, j]
                        self.grid.acceleration_body1[i, j][0] = self.grid.force_body1[i, j][0] / self.grid.mass_body1[i, j]
                        self.grid.acceleration_body1[i, j][1] = self.grid.force_body1[i, j][1] / self.grid.mass_body1[i, j]

                        if self.time <= self.dt:
                            self.grid.velocity_body1[i, j][0] += self.grid.acceleration_body1[i, j][0] * 0.5 * dt
                            self.grid.velocity_body1[i, j][1] += self.grid.acceleration_body1[i, j][1] * 0.5 * dt
                        else:
                            self.grid.velocity_body1[i, j][0] += self.grid.acceleration_body1[i, j][0] * dt
                            self.grid.velocity_body1[i, j][1] += self.grid.acceleration_body1[i, j][1] * dt

                    elif self.grid.mass_body2[i, j] > 0.0:
                        self.grid.velocity_body2[i, j][0] = self.grid.momentum_body2[i, j][0] / self.grid.mass_body2[i, j]
                        self.grid.velocity_body2[i, j][1] = self.grid.momentum_body2[i, j][1] / self.grid.mass_body2[i, j]
                        self.grid.acceleration_body2[i, j][0] = self.grid.force_body2[i, j][0] / self.grid.mass_body2[i, j]
                        self.grid.acceleration_body2[i, j][1] = self.grid.force_body2[i, j][1] / self.grid.mass_body2[i, j]

                        if self.time <= self.dt:
                            self.grid.velocity_body2[i, j][0] += self.grid.acceleration_body2[i, j][0] * 0.5 * dt
                            self.grid.velocity_body2[i, j][1] += self.grid.acceleration_body2[i, j][1] * 0.5 * dt
                        else:
                            self.grid.velocity_body2[i, j][0] += self.grid.acceleration_body2[i, j][0] * dt
                            self.grid.velocity_body2[i, j][1] += self.grid.acceleration_body2[i, j][1] * dt

    @ti.kernel
    def grid_to_particle(self):
        for p_idx in range(self.particles.num_particles):
            particle_pos = self.particles.position[p_idx]
            mp_obj_id = self.particles.object_id[p_idx]
            velocity_update = ti.Vector([0.0, 0.0])
            strain_rate = ti.Matrix.zero(ti.f32, 2, 2)
            acceleration_update = ti.Vector([0.0, 0.0])
            # normal_update = ti.Vector([0.0, 0.0])
            density_rate = 0.0
            i = int(particle_pos[0] / self.grid.cell_size)
            j = int(particle_pos[1] / self.grid.cell_size)

            for di in ti.static(range(-2, 3)):
                for dj in ti.static(range(-2, 3)):
                    cell_i = i + di
                    cell_j = j + dj

                    if 0 <= cell_i < self.grid.node_count_x and 0 <= cell_j < self.grid.node_count_y:
                        node_mass = self.grid.mass[cell_i, cell_j]

                        if node_mass > 0:
                            velocity = ti.Vector([0.0, 0.0])
                            acceleration = ti.Vector([0.0, 0.0])
                            normals = ti.Vector([0.0, 0.0])

                            node_pos = self.grid.position[cell_i, cell_j]
                            shape_x, shape_y = shape_function(particle_pos, node_pos, self.grid.cell_size)
                            grad_shape_x, grad_shape_y = gradient_shape_function(particle_pos, node_pos, self.grid.cell_size)

                            shape_value = shape_x * shape_y

                            if self.grid.body_id[cell_i * self.grid.node_count_y + cell_j] == mp_obj_id:
                                velocity[0] = self.grid.velocity_body1[cell_i, cell_j][0]
                                velocity[1] = self.grid.velocity_body1[cell_i, cell_j][1]
                                acceleration[0] = self.grid.acceleration_body1[cell_i, cell_j][0]
                                acceleration[1] = self.grid.acceleration_body1[cell_i, cell_j][1]
                                normals[0] = self.grid.normals_body1[cell_i, cell_j][0]
                                normals[1] = self.grid.normals_body1[cell_i, cell_j][1]
                            elif self.grid.body_id[cell_i * self.grid.node_count_y + cell_j + self.grid.total_nodes] == mp_obj_id:
                                velocity[0] = self.grid.velocity_body2[cell_i, cell_j][0]
                                velocity[1] = self.grid.velocity_body2[cell_i, cell_j][1]
                                acceleration[0] = self.grid.acceleration_body2[cell_i, cell_j][0]
                                acceleration[1] = self.grid.acceleration_body2[cell_i, cell_j][1]
                                normals[0] = self.grid.normals_body2[cell_i, cell_j][0]
                                normals[1] = self.grid.normals_body2[cell_i, cell_j][1]

                            velocity_update[0] += velocity[0] * shape_value
                            velocity_update[1] += velocity[1] * shape_value
                            acceleration_update[0] += acceleration[0] * shape_value
                            acceleration_update[1] += acceleration[1] * shape_value
                            # normal_update[0] += normals[0] * shape_value
                            # normal_update[1] += normals[1] * shape_value
                            density_rate -= self.particles.density[p_idx] * (
                                (grad_shape_x * shape_y * velocity[0]) + 
                                (grad_shape_y * shape_x * velocity[1])
                            )

                            # Compute strain rate
                            strain_rate[0, 0] += velocity[0] * grad_shape_x * shape_y
                            strain_rate[0, 1] += 0.5*(velocity[0] * grad_shape_y * shape_x + velocity[1] * grad_shape_x * shape_y)
                            strain_rate[1, 0] = strain_rate[0, 1]
                            strain_rate[1, 1] += velocity[1] * grad_shape_y * shape_x

            self.particles.Gvelocity[p_idx] = velocity_update
            self.particles.acceleration[p_idx] = acceleration_update
            self.particles.strain_rate[p_idx] = strain_rate
            self.particles.density_rate[p_idx] = density_rate
    @ti.kernel
    def update_particles(self):
        for p_idx in range(self.particles.num_particles):
            # Update position
            self.particles.velocity[p_idx] += self.particles.acceleration[p_idx] * self.dt
            self.particles.position[p_idx] += self.particles.Gvelocity[p_idx] * self.dt
            self.particles.density[p_idx] += self.particles.density_rate[p_idx] * self.dt
            self.particles.volume[p_idx] = self.particles.mass[p_idx] / self.particles.density[p_idx]
            # Update stress using material model
            stress_rate = self.particles.compute_stress_rate(p_idx)
            self.particles.stress[p_idx] += stress_rate * self.dt
