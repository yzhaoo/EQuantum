# self_consistent_solver.py

class FSC:
    def __init__(self, system, quantum_solver, poisson_solver, convergence_tol=1e-6, max_iter=50):
        """
        Initialize the self-consistent solver.

        Parameters:
          - system: an instance of your System class (which includes Geometry3D, Site objects, etc.)
          - quantum_solver: an instance (or module) that performs the quantum calculation (QAA update)
          - poisson_solver: an instance (or module) that performs the electrostatic/Poisson calculation (PAA update)
          - convergence_tol: tolerance for convergence.
          - max_iter: maximum number of iterations allowed.
        """
        self.system = system
        self.quantum_solver = quantum_solver
        self.poisson_solver = poisson_solver
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter

        # Import the relaxation functions from separate files.
        # These functions must be defined in their respective modules.
        # For example:
        from stepI import relax_step_I
        from stepII import relax_step_II
        from stepIII import relax_step_III
        self.relax_step_I = relax_step_I
        self.relax_step_II = relax_step_II
        self.relax_step_III = relax_step_III
        # To store metrics for convergence (e.g., previous potential or density)
        self.prev_potential = None
        self.prev_density = None

    def iterate(self):
        """
        Run the self-consistent iteration loop until convergence or until max_iter is reached.
        The loop structure follows Fig.8 of the paper:
          - Step I: Update the Q/Q' partition (remove depleted regions)
          - Step II: Relax the Poisson (PAA) update (update potential)
          - Step III: Relax the quantum (QAA) update (update ILDOS/density)
        """
        for iter_num in range(self.max_iter):
            print(f"Iteration {iter_num}")

            # Step I: Relaxation of the Q/Q' partition.
            # This function should update the system to remove depleted sites from Q'.
            self.relax_step_I(self.system)

            # Step II: Relax the Poisson approximation.
            # This function is responsible for updating the potential via the Poisson solver.
            self.relax_step_II(self.poisson_solver, self.system)

            # Step III: Relax the quantum approximation.
            # This function updates the quantum solution (e.g., recalculating the ILDOS) in the system.
            self.relax_step_III(self.quantum_solver, self.system)

            # Check for convergence.
            if self.check_convergence():
                print(f"Convergence reached at iteration {iter_num}")
                break
        else:
            print("Warning: Maximum iterations reached without full convergence.")

    def check_convergence(self):
        """
        Check convergence based on changes in the potential and/or density.
        Here we use a simple example that compares the norm of the potential change.
        You can modify this to suit the quantitative criteria from your paper.

        Returns:
          - True if the change is below the tolerance, False otherwise.
        """
        # Extract current potential from the system (e.g., from the sites)
        current_potential = [site.potential for site in self.system.sites.values()]
        current_density = [site.density for site in self.system.sites.values()]

        # If this is the first iteration, store and return False.
        if self.prev_potential is None or self.prev_density is None:
            self.prev_potential = current_potential
            self.prev_density = current_density
            return False

        # Compute norms of the differences.
        pot_diff = sum(abs(cp - pp) for cp, pp in zip(current_potential, self.prev_potential))
        dens_diff = sum(abs(cd - pd) for cd, pd in zip(current_density, self.prev_density))

        # Update previous values.
        self.prev_potential = current_potential
        self.prev_density = current_density

        # Check if both differences are below the tolerance.
        if pot_diff < self.convergence_tol and dens_diff < self.convergence_tol:
            return True
        return False
