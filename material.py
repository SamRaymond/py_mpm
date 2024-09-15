import numpy as np

class LinearElastic:
    def __init__(self, density, youngs_modulus, poisson_ratio):
        self.density = density
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        
        # Compute Lame parameters
        self.lambda_ = (youngs_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.mu = youngs_modulus / (2 * (1 + poisson_ratio))

    def compute_stress_rate(self, strain_rate):
        # Compute stress rate using Hooke's law for 2D stress state
        # Expect strain_rate as a vector [exx, eyy, exy]
        assert strain_rate.shape == (2,2), "Strain rate must be a 2x2 matrix"
        
        exx, eyy, exy, eyx = strain_rate.flatten()
        
        # Compute stress components
        factor = self.youngs_modulus / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        sxx = factor * ((1 - self.poisson_ratio) * exx + self.poisson_ratio * (eyy))
        syy = factor * ((1 - self.poisson_ratio) * eyy + self.poisson_ratio * (exx))
        sxy = self.mu * 2 * exy
        
        return np.array([[sxx, sxy], [sxy, syy]])

