# MPM Simulation

This project implements a 2D Material Point Method (MPM) simulation for elastic materials. It simulates the interaction between two circular disks using MPM techniques.

## Overview

The Material Point Method is a numerical technique used to simulate the behavior of solids, fluids, and other materials. This implementation focuses on simulating elastic materials in a 2D space.

## Key Components

1. **Main Simulation (`main.py`)**: 
   - Sets up the simulation parameters
   - Initializes the grid and particles
   - Runs the main simulation loop

2. **MPM Solver (`solver.py`)**: 
   - Implements the core MPM algorithm
   - Handles particle-to-grid and grid-to-particle transfers
   - Updates particle and grid properties each time step

3. **Particles (`particles.py`)**: 
   - Manages particle data and properties
   - Handles material assignment to particles

4. **Grid (`nodes.py`)**: 
   - Defines the background grid structure
   - Manages grid nodes and their properties

5. **Material Models (`material.py`)**: 
   - Implements material behavior (currently Linear Elastic)
   - Computes stress based on strain

6. **Shape Functions (`shape_function.py`)**: 
   - Defines shape functions for MPM interpolation
   - Includes functions for shape function gradients

7. **Visualization (`plotting.py`, `results_vis.py`)**: 
   - Provides functions to visualize the simulation results
   - Includes both static plotting and animation capabilities

8. **Results Processing (`results.py`)**: 
   - Handles saving simulation data to CSV files

9. **Particle Shapes (`particle_shapes.py`)**: 
   - Defines different particle shapes (e.g., Disk, Block)
   - Generates particles for each shape

## Key Features

- Simulation of two colliding elastic disks
- Linear elastic material model
- Particle-based representation of materials
- Background grid for computation
- Visualization of particle positions, velocities, and stresses
- Energy tracking (kinetic, elastic, and total)
- Flexible particle shape generation (currently supports Disk and Block)

## Usage

1. Set up the simulation parameters in `main.py`
2. Run `main.py` to start the simulation
3. Use `results_vis.py` to visualize the simulation results

## Customization

- Adjust material properties in `main.py`
- Modify grid size and resolution in `main.py`
- Implement new material models in `material.py`
- Add boundary conditions in `solver.py`
- Create new particle shapes in `particle_shapes.py`

## Output

The simulation generates CSV files for each time step, storing particle data including:
- Position
- Velocity
- Stress
- Strain
- Volume
- Mass
- Density
- Material properties

## Visualization

The `results_vis.py` script provides an animated visualization of:
- Particle positions
- Velocity magnitudes
- Von Mises stress
- Energy evolution over time

## Notes

- The current implementation focuses on 2D simulations
- Only linear elastic materials are implemented, but the structure allows for easy addition of other material models
- The simulation uses an explicit time integration scheme
- Particle shapes can be easily extended by subclassing the `ParticleShape` class

## Future Improvements

- Implement more advanced material models (e.g., plasticity, damage)
- Add support for 3D simulations
- Implement adaptive time-stepping
- Optimize performance for larger simulations
- Add more boundary condition options
- Implement additional particle shapes

## Dependencies

- NumPy
- Matplotlib
- tqdm (for progress bars)

## Running the Simulation

1. Ensure all dependencies are installed
2. Run `python main.py` to start the simulation
3. After the simulation completes, run `python results_vis.py <output_directory>` to visualize the results

## Visualization Options

The `results_vis.py` script now accepts command-line arguments for specifying the data directory. Use it as follows: