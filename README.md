# MPM Simulation

This project implements a 2D Material Point Method (MPM) simulation for elastic materials using [Taichi](https://taichi.graphics/). It simulates the interaction between two circular disks using MPM techniques optimized for high-performance computations.

## Overview

The Material Point Method is a numerical technique used to simulate the behavior of solids, fluids, and other materials. This implementation focuses on simulating elastic materials in a 2D space, leveraging Taichi for efficient parallel computations.

## Key Components

1. **Main Simulation (`main.py`)**:
   - Sets up the simulation parameters.
   - Initializes the grid and particles.
   - Runs the main simulation loop.

2. **MPM Solver (`solver.py`)**:
   - Implements the core MPM algorithm.
   - Handles particle-to-grid and grid-to-particle transfers.
   - Updates particle and grid properties each time step.

3. **Particles (`particles.py`)**:
   - Manages particle data and properties.
   - Handles material assignment to particles.

4. **Grid (`nodes.py`)**:
   - Defines the background grid structure.
   - Manages grid nodes and their properties.

5. **Material Models (`material.py`)**:
   - Implements material behavior (currently Linear Elastic).
   - Computes stress based on strain.

6. **Shape Functions (`shape_function.py`)**:
   - Defines shape functions for MPM interpolation.
   - Includes functions for shape function gradients.

7. **Visualization (`plotting.py`, `results_vis.py`)**:
   - Provides functions to visualize the simulation results.
   - Includes both static plotting and animation capabilities.

8. **Results Processing (`results.py`)**:
   - Handles saving simulation data to CSV files.

9. **Particle Shapes (`particle_shapes.py`)**:
   - Defines different particle shapes (e.g., Disk, Block).
   - Generates particles for each shape.

## Key Features

- **Optimized Performance with Taichi**: Utilizes Taichi's parallel computing capabilities for efficient simulations.
- Simulation of two colliding elastic disks.
- Linear elastic material model.
- Particle-based representation of materials.
- Background grid for computation.
- Visualization of particle positions, velocities, and stresses.
- Energy tracking (kinetic, elastic, and total).
- Flexible particle shape generation (currently supports Disk and Block).

## Usage

1. **Set Up the Simulation Parameters**:
   - Configure simulation parameters in `main.py`, including grid size, cell size, material properties, and initial conditions.

2. **Run the Simulation**:
   ```bash
   python main.py
   ```

3. **Visualize the Results**:
   ```bash
   python results_vis.py <output_directory>
   ```

## Customization

- **Adjust Material Properties**:
  - Modify material properties in `main.py` or extend `material.py` to implement new material models.

- **Modify Grid Size and Resolution**:
  - Change `grid_size` and `cell_size` in `main.py` to alter the simulation grid.

- **Implement New Material Models**:
  - Extend `material.py` with additional material behaviors as needed.

- **Add Boundary Conditions**:
  - Update `solver.py` to incorporate new boundary conditions.

- **Create New Particle Shapes**:
  - Extend `particle_shapes.py` by subclassing `ParticleShape` to define new shapes.

## Output

The simulation generates CSV files for each time step, storing particle data including:

- **Position**: `pos_x`, `pos_y`
- **Velocity**: `vel_x`, `vel_y`
- **Stress**: `stress_xx`, `stress_xy`, `stress_yx`, `stress_yy`
- **Strain**: `strain_xx`, `strain_xy`, `strain_yx`, `strain_yy`
- **Volume**
- **Mass**
- **Density**
- **Material Properties**: `youngs_modulus`, `poisson_ratio`

## Visualization

The `results_vis.py` script provides an animated visualization of:

- Particle positions.
- Velocity magnitudes.
- Von Mises stress.
- Energy evolution over time.

### Visualization Options

The `results_vis.py` script accepts command-line arguments for specifying the data directory. Use it as follows:

## Notes

- The current implementation focuses on 2D simulations.
- Only linear elastic materials are implemented, but the structure allows for easy addition of other material models.
- The simulation uses an explicit time integration scheme.
- Particle shapes can be easily extended by subclassing the `ParticleShape` class.
- **Taichi Integration**: All computationally intensive operations are optimized using Taichi, ensuring high performance and scalability.

## Future Improvements

- Implement more advanced material models (e.g., plasticity, damage).
- Add support for 3D simulations.
- Implement adaptive time-stepping.
- Optimize performance for larger simulations.
- Add more boundary condition options.
- Implement additional particle shapes.

## Dependencies

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://tqdm.github.io/) (for progress bars)
- [Taichi](https://taichi.graphics/)

## Running the Simulation

1. **Ensure All Dependencies Are Installed**:
   ```bash
   pip install numpy matplotlib tqdm taichi
   ```

2. **Start the Simulation**:
   ```bash
   python main.py
   ```

3. **Visualize the Results After Completion**:
   ```bash
   python results_vis.py simulation_output_taichi
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.