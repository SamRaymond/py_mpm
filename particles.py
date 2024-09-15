import numpy as np
from material import LinearElastic  # Import your material models

class Particles:
    def __init__(self, particle_data, material_properties):
        self.particles = []
        self.material_properties = material_properties
        print(f"Initializing Particles with {len(particle_data)} particle data")
        self.create_particles(particle_data)
        print(f"Finished creating {len(self.particles)} particles")

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

                particle = {
                    'position': position,
                    'velocity': velocity,
                    'density': material_props['density'],
                    'mass': data['mass'],
                    'volume': data['volume'],
                    'stress': np.zeros((2, 2)),
                    'strain': np.zeros((2, 2)),
                    'id': data.get('id', len(self.particles)),
                    'object_id': object_id,
                    'material': self.create_material(material_props)
                }
                self.particles.append(particle)
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
        return len(self.particles)

    def assign_material(self, material_type, properties):
        material = {
            "type": material_type,
            "properties": properties
        }
        return material

    def get_particles(self):
        return self.particles
    
    def get_particle(self, index):
        return self.particles[index]    
    
    def compute_dt(self):
        # Compute the time step using CFL condition with stiffness
        cfl_factor = 0.2
        max_wave_speed = 0.0
        min_cell_size = float('inf')

        for particle in self.particles:
            # Calculate the wave speed for each particle
            if 'material' in particle and 'properties' in particle['material']:
                properties = particle['material']['properties']
                if 'youngs_modulus' in properties and 'density' in properties:
                    E = properties['youngs_modulus']
                    rho = properties['density']
                    wave_speed = np.sqrt(E / rho)
                    max_wave_speed = max(max_wave_speed, wave_speed)

            # Find the minimum cell size
            if 'cell_size' in particle:
                min_cell_size = min(min_cell_size, particle['cell_size'])

        # If cell_size is not available in particle data, use a default value
        if min_cell_size == float('inf'):
            min_cell_size = 0.05  # Default value, adjust as needed

        # Compute dt using CFL condition with stiffness
        if max_wave_speed > 0:
            dt = cfl_factor * min_cell_size / max_wave_speed
        else:
            # Fallback if no valid wave speed could be calculated
            dt = cfl_factor * min_cell_size

        return dt
    
    def set_object_velocity(particles, object_id, velocity):
        for particle in particles.particles:
            if particle['object_id'] == object_id:
                particle['velocity'] = np.array(velocity)

    def assign_material_properties(self, material, object_id):
        for particle in self.particles:
            if particle['object_id'] == object_id:
                particle['material'] = self.create_material(material)
                particle['density'] = material['density']
                particle['volume'] = particle['mass'] / particle['density']

