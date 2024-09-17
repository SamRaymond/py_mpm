import numpy as np

class LinearElastic:
    def __init__(self, density, youngs_modulus, poisson_ratio):
        self.density = density
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio

    def compute_stress_rate(self, strain_rate):
        assert strain_rate.shape == (2,2), "Strain rate must be a 2x2 matrix"
        
        # Calculate bulk modulus K and shear modulus G from Young's modulus and Poisson's ratio
        K = self.youngs_modulus / (3 * (1 - 2 * self.poisson_ratio))
        G = self.youngs_modulus / (2 * (1 + self.poisson_ratio))
        
        # Calculate volumetric strain rate
        volumetric_strain_rate = np.trace(strain_rate)
        
        # Calculate deviatoric strain rate
        deviatoric_strain_rate = strain_rate - (volumetric_strain_rate / 3) * np.eye(2)
        
        # Compute stress rate using volumetric and deviatoric parts
        volumetric_stress_rate = 3 * K * volumetric_strain_rate * np.eye(2)
        deviatoric_stress_rate = 2 * G * deviatoric_strain_rate
        
        stress_rate = volumetric_stress_rate + deviatoric_stress_rate
        
        return stress_rate

