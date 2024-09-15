import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_particles_and_grid(particles, grid, step):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Extract positions and velocities
    positions = np.array([p['position'] for p in particles.get_particles()])
    velocities = np.array([p['velocity'] for p in particles.get_particles()])
    
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
    fig, ax = plt.subplots(figsize=(5, 5))
    # Extract positions and object IDs
    positions = np.array([p['position'] for p in particles.get_particles()])
    object_ids = np.array([p['object_id'] for p in particles.get_particles()])
    
    print(f"Total particles: {len(positions)}")
    print(f"Unique object IDs: {np.unique(object_ids)}")
    # print(f"Position range: x({positions[:, 0].min()}, {positions[:, 0].max()}), y({positions[:, 1].min()}, {positions[:, 1].max()})")
    
    # Create a color map for different object IDs
    unique_ids = np.unique(object_ids)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_ids)))
    color_map = dict(zip(unique_ids, colors))
    
    # Draw particles as squares
    square_size = grid.cell_size / 2
    for pos, obj_id in zip(positions, object_ids):
        color = color_map[obj_id]
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
    ax.set_title('Particles with Object ID')
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=color_map[id], edgecolor='none', label=f'Object {id}') 
                       for id in unique_ids]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()