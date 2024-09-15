import csv
import numpy as np

def save_particles_to_csv(particles, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow([
            'particle_id', 'object_id', 'pos_x', 'pos_y', 'vel_x', 'vel_y',
            'stress_xx', 'stress_xy', 'stress_yx', 'stress_yy',
            'strain_xx', 'strain_xy', 'strain_yx', 'strain_yy',
            'volume', 'mass', 'density',
            'youngs_modulus', 'poisson_ratio'
        ])
        
        # Write particle data
        for p_idx in range(particles.get_particle_count()):
            particle = particles.get_particle(p_idx)
            pos = particle['position']
            vel = particle['velocity']
            stress = particle['stress']
            strain = particle['strain']
            material = particle['material']
            
            csv_writer.writerow([
                p_idx,
                particle['object_id'],
                pos[0], pos[1],
                vel[0], vel[1],
                stress[0,0], stress[0,1], stress[1,0], stress[1,1],
                strain[0,0], strain[0,1], strain[1,0], strain[1,1],
                particle['volume'],
                particle['mass'],
                particle['density'],
                material.youngs_modulus,
                material.poisson_ratio
            ])

