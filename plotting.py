import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_particles_and_grid(particles, grid, step):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Extract positions and velocities
    positions = particles.position
    velocities = particles.velocity
    
    # Ensure velocities is 2D
    if velocities.ndim == 1:
        velocities = velocities.reshape(-1, 1)
    
    # Calculate velocity magnitudes
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    # Create color map for particles
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=velocity_magnitudes.min(), vmax=velocity_magnitudes.max())
    
    # Draw particles as squares
    square_size = grid.cell_size / 2
    for pos, vel_mag in zip(positions, velocity_magnitudes):
        color = cmap(norm(vel_mag))
        square = patches.Rectangle((pos[0] - square_size/2, pos[1] - square_size/2), 
                                   square_size, square_size, 
                                   facecolor=color, edgecolor='none', alpha=0.8)
        ax.add_patch(square)
    
    # Draw MPM grid
    for i in range(grid.node_count[0] + 1):
        ax.axvline(x=i * grid.cell_size, color='gray', linestyle=':', alpha=0.5)
    for j in range(grid.node_count[1] + 1):
        ax.axhline(y=j * grid.cell_size, color='gray', linestyle=':', alpha=0.5)
    
    # Set plot properties
    ax.set_aspect('equal')
    ax.set_xlim(0, grid.physical_size[0])
    ax.set_ylim(0, grid.physical_size[1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Particles and MPM Grid - Step {step}')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Velocity Magnitude')
    
    plt.tight_layout()
    plt.show()

def visualize_particles_with_object_id(particles, grid):
    fig, ax = plt.subplots()
    unique_ids = np.unique(particles.object_ids)
    color_map = plt.cm.get_cmap('tab20', len(unique_ids))
    
    for obj_id in unique_ids:
        mask = particles.object_ids == obj_id
        ax.scatter(particles.positions[mask, 0], particles.positions[mask, 1], 
                   label=f'Object {obj_id}', color=color_map(obj_id))
    
    ax.set_xlim(0, grid.physical_size[0])
    ax.set_ylim(0, grid.physical_size[1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Particles with Object ID')
    
    # Add grid lines
    ax.grid(True)
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=color_map(id), edgecolor='none', label=f'Object {id}') 
                       for id in unique_ids]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()