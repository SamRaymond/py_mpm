import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import argparse

# Add command-line argument parsing
parser = argparse.ArgumentParser(description="Visualize simulation results.")
parser.add_argument("data_dir", help="Directory containing the CSV files")
args = parser.parse_args()

# Use the provided data directory
data_dir = args.data_dir

# Variables to store grid boundaries
grid_min_x = float('inf')
grid_max_x = float('-inf')
grid_min_y = float('inf')
grid_max_y = float('-inf')

# Function to read a CSV file
def read_csv(filename):
    global grid_min_x, grid_max_x, grid_min_y, grid_max_y
    with open(os.path.join(data_dir, filename), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header
        data = [list(map(float, row)) for row in reader if row]  # Convert to float and skip empty rows
    if not data:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    data = np.array(data)
    positions = data[:, 2:4]
    velocities = data[:, 4:6]
    stress = data[:, 6:10]
    volumes = data[:, 14]
    masses = data[:, 15]
    
    # Update grid boundaries
    grid_min_x = min(grid_min_x, positions[:, 0].min())
    grid_max_x = max(grid_max_x, positions[:, 0].max())
    grid_min_y = min(grid_min_y, positions[:, 1].min())
    grid_max_y = max(grid_max_y, positions[:, 1].max())
    
    return positions, velocities, stress, volumes, masses

# Read the first file to get the number of particles and initialize the plot
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
initial_positions, initial_velocities, initial_stress, initial_volumes, initial_masses = read_csv(csv_files[0])
num_particles = len(initial_positions)

# Create the figure and axes
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

scatter1 = ax1.scatter([], [], s=60, c=[], cmap='viridis', marker='s')
scatter2 = ax2.scatter([], [], s=60, c=[], cmap='plasma', marker='s')
line1, = ax3.plot([], [], label='Kinetic Energy')
line2, = ax3.plot([], [], label='Elastic Energy')
line3, = ax3.plot([], [], label='Total Energy')

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.set_title("Velocity Magnitude")
ax2.set_title("Von Mises Stress")
ax3.set_title("Energy Evolution")
ax3.set_xlabel("Step")
ax3.set_ylabel("Energy")
ax3.legend()

title = fig.suptitle("Step: 0")

# Add colorbars
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar1.set_label('Velocity magnitude')
cbar2.set_label('Von Mises stress')

# Initialize the color normalizations
norm1 = Normalize()
norm2 = Normalize()

# Lists to store energy values and stress range
kinetic_energy = []
elastic_energy = []
total_energy = []
stress_min = float('inf')
stress_max = float('-inf')

# Function to calculate von Mises stress
def von_mises_stress(stress):
    xx, xy, yx, yy = stress.T
    return np.sqrt(0.5 * ((xx - yy)**2 + (yy - xx)**2 + 6 * (xy**2)))

# Function to calculate kinetic energy
def calculate_kinetic_energy(velocities, masses):
    return 0.5 * np.sum(masses[:, np.newaxis] * np.sum(velocities**2, axis=1))

# Function to calculate elastic energy
def calculate_elastic_energy(stress, volumes, youngs_modulus):
    return 0.5 * np.sum(volumes[:, np.newaxis] * np.sum(stress**2, axis=1)) / youngs_modulus

# Animation update function
def update(frame):
    global kinetic_energy, elastic_energy, total_energy, stress_min, stress_max
    
    # Reset energy lists and stress range if the animation is starting over
    if frame == 0:
        kinetic_energy = []
        elastic_energy = []
        total_energy = []
        stress_min = float('inf')
        stress_max = float('-inf')
    
    positions, velocities, stress, volumes, masses = read_csv(csv_files[frame])
    if len(positions) == 0:
        return scatter1, scatter2, line1, line2, line3, title
    
    scatter1.set_offsets(positions)
    scatter2.set_offsets(positions)
    
    # Calculate velocity magnitudes
    vel_mag = np.linalg.norm(velocities, axis=1)
    
    # Calculate von Mises stress
    von_mises = von_mises_stress(stress)
    
    # Update stress range
    stress_min = min(stress_min, von_mises.min())
    stress_max = max(stress_max, von_mises.max())
    
    # Update color normalizations
    # norm1.autoscale(vel_mag)
    # norm1.autoscale(velocities[:, 0])
    norm1.vmin = -1.0#vel_mag.min()
    norm1.vmax = 1.0
    norm2.autoscale(von_mises)
    # norm2.vmin = 0.0#stress_min
    # norm2.vmax = 1.0e5#stress_max
    
    # scatter1.set_array(vel_mag)
    scatter1.set_array(velocities[:, 0])
    scatter2.set_array(von_mises)
    
    # Update colorbars
    scatter1.set_norm(norm1)
    scatter2.set_norm(norm2)
    cbar1.update_normal(scatter1)
    cbar2.update_normal(scatter2)
    
    # Calculate and store energies
    youngs_modulus = float(csv_files[frame].split('_')[-1].split('.')[0])  # Extract Young's modulus from filename
    ke = calculate_kinetic_energy(velocities, masses)
    ee = 1e-4*calculate_elastic_energy(stress, volumes, youngs_modulus)
    te = ke + ee
    kinetic_energy.append(ke)
    elastic_energy.append(ee)
    total_energy.append(te)
    
    # Update energy plots
    steps = range(len(kinetic_energy))
    line1.set_data(steps, kinetic_energy)
    line2.set_data(steps, elastic_energy)
    line3.set_data(steps, total_energy)
    ax3.relim()
    ax3.autoscale_view()
    
    # Update axes limits
    ax1.set_xlim(grid_min_x, grid_max_x)
    ax1.set_ylim(grid_min_y, grid_max_y)
    ax2.set_xlim(grid_min_x, grid_max_x)
    ax2.set_ylim(grid_min_y, grid_max_y)
    
    title.set_text(f"Step: {frame}")
    return scatter1, scatter2, line1, line2, line3, title

# Create the animation
anim = FuncAnimation(fig, update, frames=len(csv_files), interval=50, blit=False)

# Save the animation (optional)
# anim.save('simulation.mp4', writer='ffmpeg', fps=30)

# Show the plot
plt.tight_layout()
plt.show()