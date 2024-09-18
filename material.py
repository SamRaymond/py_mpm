import taichi as ti

# Define material properties as Taichi data-oriented classes
@ti.data_oriented
class LinearElastic:
    def __init__(self, youngs_modulus, poisson_ratio, density):
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.density = density

        # Precompute bulk modulus K and shear modulus G
        self.K = self.E / (3 * (1 - 2 * self.nu))
        self.G = self.E / (2 * (1 + self.nu))

    @ti.func
    def compute_stress_rate(self, strain_rate):
        # strain_rate: 2x2 ti.Matrix
        volumetric_strain_rate = strain_rate.trace()
        deviatoric_strain_rate = strain_rate - (volumetric_strain_rate / 3.0) * ti.Matrix.identity(ti.f32, 2)

        # Compute stress rate using volumetric and deviatoric parts
        volumetric_stress_rate = 3.0 * self.K * volumetric_strain_rate * ti.Matrix.identity(ti.f32, 2)
        deviatoric_stress_rate = 2.0 * self.G * deviatoric_strain_rate

        stress_rate = volumetric_stress_rate + deviatoric_stress_rate

        return stress_rate

