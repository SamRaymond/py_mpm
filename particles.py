import numpy as np
from material import LinearElastic  # Import your material models

class Particles:
    def __init__(self, particle_data, material_properties,cell_size):
        self.material_properties = material_properties
        self.num_particles = len(particle_data)
        self.cell_size = cell_size
        
        # Initialize numpy arrays for particle properties
        self.position = np.zeros((self.num_particles, 2))
        self.velocity = np.zeros((self.num_particles, 2))
        self.Gvelocity = np.zeros((self.num_particles, 2))
        self.acceleration = np.zeros((self.num_particles, 2))
        self.strain_rate = np.zeros((self.num_particles, 2, 2))
        self.density = np.zeros(self.num_particles)
        self.mass = np.zeros(self.num_particles)
        self.volume = np.zeros(self.num_particles)
        self.density_rate = np.zeros((self.num_particles))
        self.stress = np.zeros((self.num_particles, 2, 2))
        self.strain = np.zeros((self.num_particles, 2, 2))
        self.ids = np.zeros(self.num_particles, dtype=int)
        self.object_id = np.zeros(self.num_particles, dtype=int)
        self.normals = np.zeros((self.num_particles, 2))
        self.materials = [None] * self.num_particles

        print(f"Initializing Particles with {self.num_particles} particle data")
        self.create_particles(particle_data)
        print(f"Finished creating {self.num_particles} particles")

    def create_particles(self, particle_data):
        for i, data in enumerate(particle_data):
            try:
                position = np.array(data['position'], dtype=float)
                velocity = np.array(data.get('velocity', [0.0, 0.0]), dtype=float)
                
                # Ensure position and velocity are 2D vectors
                if position.shape != (2,):
                    raise ValueError(f"Position must be a 2D vector. Got shape {position.shape}")
                if velocity.shape != (2,):
                    raise ValueError(f"Velocity must be a 2D vector. Got shape {velocity.shape}")

                object_id = data['object_id']
                material_props = self.material_properties[object_id]

                self.position[i] = position
                self.velocity[i] = velocity
                self.density[i] = material_props['density']
                self.mass[i] = data['mass']
                self.volume[i] = data['mass'] / material_props['density']
                self.ids[i] = data.get('id', i)
                self.normals[i] = data.get('normals', [0.0, 0.0])
                self.object_id[i] = object_id
                self.materials[i] = self.create_material(material_props)
            except Exception as e:
                print(f"Error creating particle {i}: {str(e)}")
                print(f"Particle data: {data}")
                print(f"Material properties: {material_props}")

    def create_material(self, properties):
        if properties['type'] == "LinearElastic":
            return LinearElastic(
                youngs_modulus=properties['youngs_modulus'],
                poisson_ratio=properties['poisson_ratio'],
                density=properties['density']
            )
        else:
            raise ValueError(f"Unknown material type: {properties['type']}")

    def get_particle_count(self):
        return self.num_particles

    def get_particles(self):
        return {
            'position': self.position,
            'velocity': self.velocity,
            'Gvelocity': self.Gvelocity,
            'accelerations': self.accelerations,
            'strain_rate': self.strain_rate,
            'densities': self.densities,
            'masses': self.masses,
            'volume': self.volume,
            'density_rate': self.density_rate,
            'stress': self.stress,
            'strain': self.strain,
            'normals': self.normals,
            'ids': self.ids,
            'object_id': self.object_id,
            'materials': self.materials
        }

    def get_particle(self, index):
        return {
            'position': self.position[index],
            'velocity': self.velocity[index],
            'Gvelocity': self.Gvelocity[index],
            'acceleration': self.acceleration[index],
            'strain_rate': self.strain_rate[index],
            'density': self.density[index],
            'mass': self.mass[index],
            'volume': self.volume[index],
            'density_rate': self.density_rate[index],
            'stress': self.stress[index],
            'strain': self.strain[index],
            'normals': self.normals[index],
            'id': self.ids[index],
            'object_id': self.object_id[index],
            'material': self.materials[index]
        }

    def compute_dt(self,cfl_factor=0.2):
        # Compute the time step using CFL condition with stiffness
        max_wave_speed = 0.0
        min_cell_size = float('inf')

        for i in range(self.num_particles):
            material = self.materials[i]
            if material:
                E = material.youngs_modulus
                rho = material.density
                wave_speed = np.sqrt(E / rho)
                max_wave_speed = max(max_wave_speed, wave_speed)

            # Find the minimum cell size
            # Assuming cell_size is a property of the particle, which is not defined in the original code
            # Replace with actual logic to determine cell size if available
            min_cell_size = min(min_cell_size, self.cell_size)

        # Compute dt using CFL condition with stiffness
        if max_wave_speed > 0:
            dt = cfl_factor * min_cell_size / max_wave_speed
        else:
            # Fallback if no valid wave speed could be calculated
            dt = cfl_factor * min_cell_size

        return dt

    def set_object_velocity(self, object_id, velocity):
        for i in range(self.num_particles):
            if self.object_id[i] == object_id:
                self.velocity[i] = np.array(velocity)

    def assign_material_properties(self, material, object_id):
        for i in range(self.num_particles):
            if self.object_id[i] == object_id:
                self.materials[i] = self.create_material(material)
                self.density[i] = material['density']
                self.volume[i] = self.mass[i] / self.density[i]


    def reset_particles(self):
        self.acceleration = np.zeros((self.num_particles, 2))
        self.strain_rate = np.zeros((self.num_particles, 2, 2))
        self.density_rate = np.zeros((self.num_particles))
        # self.normals = np.zeros((self.num_particles, 2))

