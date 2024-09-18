import os
import csv
import numpy as np
from tqdm import tqdm
from particle_shapes import Disk
from nodes import Grid
from particles import Particles
from solver import MPMSolver
from plotting import visualize_particles_and_grid
from results import save_particles_to_csv
# import projections

# Setup grid parameters
grid_size = (0.5, 0.5)  # Adjust as needed
cell_size = 0.01  # Adjust based on your simulation scale

# Initialize the grid
grid = Grid(grid_size, cell_size)

# Calculate the physical size of the grid
physical_size = (grid_size[0] * cell_size, grid_size[1] * cell_size)

print(f"Grid initialized with size {grid_size} and physical dimensions {physical_size}")
print(f"Total number of nodes: {grid.total_nodes}")  # We now have 2500 nodes (50 * 50)

# Nodes can be accessed using grid.nodes
# For example, to access the node at position (i, j):
# node = grid.get_node(i, j)


object_ids = [1,2]

# Initialize Material Points
# Define material properties
material_properties = {
    object_ids[0]: {  # Object ID 1
        "type": "LinearElastic",
        "density": 1000,  # kg/m^3
        "youngs_modulus": 1e8,  # Pa
        "poisson_ratio": 0.3
    },
    object_ids[1]: {  # Object ID 2
        "type": "LinearElastic",
        "density": 1000,  # kg/m^3
        "youngs_modulus": 1e8,  # Pa
        "poisson_ratio": 0.3
    }
}

# Define disk parameters
disk1_radius = 6*cell_size
disk1_center = (0.25, 0.25)
disk1_object_id = 1

disk2_radius = 6*cell_size
# Calculate the center of the second disk
# It should be 2 cells from the edge of the first disk
disk2_x = disk1_center[0] + disk1_radius + 3*cell_size + disk2_radius
disk2_center = (disk2_x, disk1_center[1])
disk2_object_id = 2

# Create Disk objects
disk1 = Disk(cell_size, disk1_radius, disk1_center, object_ids[0], material_properties[object_ids[0]]["density"])
disk2 = Disk(cell_size, disk2_radius, disk2_center, object_ids[1], material_properties[object_ids[1]]["density"])

# Generate particles for both disks
particles1 = disk1.generate_particles()
particles2 = disk2.generate_particles()
combined_particles = particles1 + particles2



# Create Particles object
particles = Particles(combined_particles, material_properties,cell_size)

# Set initial velocities for each object
particles.set_object_velocity(object_id=object_ids[0], velocity=[1.0, 0.0])  # Object 1 moving right
particles.set_object_velocity(object_id=object_ids[1], velocity=[-1.0, 0.0])  # Object 2 moving left

print(f"Generated {len(particles1)} particles for disk 1")
print(f"Generated {len(particles2)} particles for disk 2")
print(f"Total particles: {particles.get_particle_count()}")

# # # Call the visualization function
# visualize_particles_and_grid(particles, grid, 0)

# # Wait for keyboard input to continue or exit
# while True:
#     user_input = input("Press Enter to continue or '0' to exit: ")
#     if user_input == '0':
#         print("Exiting simulation...")
#         exit()
#     elif user_input == '':
#         print("Continuing simulation...")
#         break
#     else:
#         print("Invalid input. Please press Enter to continue or '0' to exit.")

dt = particles.compute_dt(cfl_factor=0.2)

print(f"dt: {dt}")
# Create MPM solver

solver = MPMSolver(particles, grid, dt)

num_steps = 10000

# Create an output directory if it doesn't exist
output_dir = "simulation_output"
os.makedirs(output_dir, exist_ok=True)

# Main simulation loop
for step in tqdm(range(num_steps), desc="Simulating"):
    solver.step()
    
    # Save outputs every 100 steps
    if step % 100 == 0:
        output_file = os.path.join(output_dir, f"step_{step:05d}.csv")
        save_particles_to_csv(particles,output_file)
        print(f"Saved output for step {step} to {output_file}")

print("Simulation completed successfully!")
