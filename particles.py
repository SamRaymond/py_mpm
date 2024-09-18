import taichi as ti
import numpy as np
from material import LinearElastic  # Import your material models

@ti.data_oriented
class Particles:
    def __init__(self, particle_data, material_properties, cell_size):
        self.material_properties = material_properties
        self.num_particles = len(particle_data)
        self.cell_size = cell_size

        # Initialize Taichi fields for particle properties
        self.position = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.velocity = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.Gvelocity = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.acceleration = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        self.strain_rate = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_particles)
        self.density = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.mass = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.volume = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.density_rate = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.stress = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_particles)
        self.strain = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.num_particles)
        self.ids = ti.field(dtype=ti.i32, shape=self.num_particles)
        self.object_id = ti.field(dtype=ti.i32, shape=self.num_particles)
        self.normals = ti.Vector.field(2, dtype=ti.f32, shape=self.num_particles)
        # For materials, we'll create per-particle material properties fields
        self.youngs_modulus = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.poisson_ratio = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.material_density = ti.field(dtype=ti.f32, shape=self.num_particles)
        # We'll store K and G for each particle
        self.K = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.G = ti.field(dtype=ti.f32, shape=self.num_particles)

        # Initialize the particles
        self.create_particles(particle_data)
        print(f"Initialized {self.num_particles} particles.")

    def create_particles(self, particle_data):
        # Temporary Python lists to hold data before transferring to Taichi fields
        position_list = []
        velocity_list = []
        normals_list = []
        for i, data in enumerate(particle_data):
            try:
                position = ti.Vector(data['position'])
                velocity = ti.Vector(data.get('velocity', [0.0, 0.0]))
                object_id = data['object_id']
                material_props = self.material_properties[object_id]

                # Append to lists
                position_list.append(position)
                velocity_list.append(velocity)
                normals_list.append(ti.Vector(data.get('normals', [0.0, 0.0])))

                # Set scalar properties
                self.density[i] = material_props['density']
                self.mass[i] = data['mass']
                self.volume[i] = data['mass'] / material_props['density']
                self.ids[i] = data.get('id', i)
                self.object_id[i] = object_id
                self.youngs_modulus[i] = material_props['youngs_modulus']
                self.poisson_ratio[i] = material_props['poisson_ratio']
                self.material_density[i] = material_props['density']
                # Compute K and G
                self.K[i] = self.youngs_modulus[i] / (3.0 * (1.0 - 2.0 * self.poisson_ratio[i]))
                self.G[i] = self.youngs_modulus[i] / (2.0 * (1.0 + self.poisson_ratio[i]))
            except Exception as e:
                print(f"Error creating particle {i}: {str(e)}")
                print(f"Particle data: {data}")
                print(f"Material properties: {material_props}")

        # Transfer lists to Taichi fields
        self.position.from_numpy(np.array([p.to_numpy() for p in position_list], dtype=np.float32))
        self.velocity.from_numpy(np.array([v.to_numpy() for v in velocity_list], dtype=np.float32))
        self.normals.from_numpy(np.array([n.to_numpy() for n in normals_list], dtype=np.float32))

    @ti.kernel
    def set_object_velocity(self, object_id: ti.i32, vx: ti.f32, vy: ti.f32):
        for i in range(self.num_particles):
            if self.object_id[i] == object_id:
                self.velocity[i] = ti.Vector([vx, vy])

    @ti.func
    def compute_stress_rate(self, p_idx):
        strain_rate = self.strain_rate[p_idx]
        volumetric_strain_rate = strain_rate.trace()
        deviatoric_strain_rate = strain_rate - (volumetric_strain_rate / 3.0) * ti.Matrix.identity(ti.f32, 2)
        volumetric_stress_rate = 3.0 * self.K[p_idx] * volumetric_strain_rate * ti.Matrix.identity(ti.f32, 2)
        deviatoric_stress_rate = 2.0 * self.G[p_idx] * deviatoric_strain_rate
        stress_rate = volumetric_stress_rate + deviatoric_stress_rate
        return stress_rate

    def get_particle_count(self):
        return self.num_particles

    def get_particle(self, index):
        # Accessing individual particle data and converting to Python types
        return {
            'position': self.position[index].to_numpy(),
            'velocity': self.velocity[index].to_numpy(),
            'Gvelocity': self.Gvelocity[index].to_numpy(),
            'acceleration': self.acceleration[index].to_numpy(),
            'strain_rate': self.strain_rate[index].to_numpy(),
            'density': float(self.density[index]),
            'mass': float(self.mass[index]),
            'volume': float(self.volume[index]),
            'density_rate': float(self.density_rate[index]),
            'stress': self.stress[index].to_numpy(),
            'strain': self.strain[index].to_numpy(),
            'normals': self.normals[index].to_numpy(),
            'id': int(self.ids[index]),
            'object_id': int(self.object_id[index]),
            'youngs_modulus': float(self.youngs_modulus[index]),
            'poisson_ratio': float(self.poisson_ratio[index]),
            'material_density': float(self.material_density[index])
        }

    def compute_dt(self, cfl_factor=0.2):
        # Compute the time step using CFL condition with stiffness
        max_wave_speed = 0.0
        min_cell_size = self.cell_size

        youngs_modulus_np = self.youngs_modulus.to_numpy()
        density_np = self.material_density.to_numpy()
        wave_speeds = np.sqrt(youngs_modulus_np / density_np)
        max_wave_speed = np.max(wave_speeds)

        # Compute dt using CFL condition with stiffness
        if max_wave_speed > 0:
            dt = cfl_factor * min_cell_size / max_wave_speed
        else:
            dt = cfl_factor * min_cell_size

        return dt

    def reset_particles(self):
        # Reset particle properties before each simulation step
        self.acceleration.fill(0.0)
        self.strain_rate.fill(0.0)
        self.density_rate.fill(0.0)
        # self.normals.fill(0.0)  # Uncomment if normals need to be reset