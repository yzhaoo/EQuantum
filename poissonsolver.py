import numpy as np
from scipy.sparse import lil_matrix, identity, hstack, vstack, spsolve
import mumps

class PoissonSolver:
    def __init__(self, sites):
        """
        Initialize the Poisson solver.

        Parameters:
        - sites: dict, mapping site IDs to Site objects.
                 Each Site must have attributes:
                   - coordinates (np.array),
                   - charge,
                   - potential (for Dirichlet sites, the prescribed value),
                   - dielectric_constant,
                   - BCtype ("n" for Neumann, "d" for Dirichlet),
                   - neighbors (dict mapping neighbor site IDs -> face area)
        """
        self.sites = sites
        self.num_sites = len(sites)
        self.A_mixed = None  # The assembled mixed matrix.
        self.F_input = None  # The input (RHS) vector in Eq. (20).
        self.N_indices = []  # Indices for Neumann sites.
        self.D_indices = []  # Indices for Dirichlet sites.


        self.Delta_matrix= None # Matrices for discrete laplacian
        self.calculate_delta() # Initialize Delta_matrix

    

    def calculate_delta(self):

        total = self.num_sites

        # Partition sites by boundary condition type.
        self.N_indices = [i for i, site in self.sites.items() if site.BCtype == "n"]
        self.D_indices = [i for i, site in self.sites.items() if site.BCtype == "d"]

        # Assemble the full finite-volume matrix Δ and full source vector b.
        A_full = lil_matrix((total, total))
        b_full = np.zeros(total)
        for i, site in self.sites.items():
            diag_val = 0.0
            for j, face_area in site.neighbors.items():
                neighbor = self.sites[j]
                d = np.linalg.norm(site.coordinates - neighbor.coordinates)
                avg_dielectric = 2.0 /(1/site.dielectric_constant + 1/neighbor.dielectric_constant)
                coeff = avg_dielectric * face_area / d
                diag_val += coeff
                A_full[i, j] = -coeff
            A_full[i, i] = diag_val
            # b_full represents the source term (e.g., the charge) at each site.
            b_full[i] = site.charge

        A_full = A_full.tocsr()

        # Partition A_full into blocks corresponding to Neumann (N) and Dirichlet (D) sites.
        A_NN = A_full[self.N_indices, :][:, self.N_indices]
        A_ND = A_full[self.N_indices, :][:, self.D_indices]
        A_DN = A_full[self.D_indices, :][:, self.N_indices]
        A_DD = A_full[self.D_indices, :][:, self.D_indices]
        #update the delta matrix
        self.Delta_matrix=[A_NN,A_ND,A_DN,A_DD]

        # define A_mixed matrix
        zero_block = lil_matrix((len(self.N_indices), len(self.D_indices)))
        top_block = hstack([A_NN, zero_block])
        # For Dirichlet sites: row = [ Δ_DN   -I ]
        I_D = identity(len(self.D_indices), format='lil')
        bottom_block = hstack([A_DN, -I_D])
        #update the A_mixied
        self.A_mixed = vstack([top_block, bottom_block]).tocsr()

    def assemble_input(self, n_N, U_D):
        [_,A_ND,_,A_DD]=self.Delta_matrix
        # Now, construct the input vector F_input.
        # For Neumann sites: F_top = n_N - Δ_ND * U_D,
        # where n_N are the source terms for Neumann sites and U_D are the prescribed potentials.
        A_ND = A_ND.tocsr()
        F_top = n_N - A_ND.dot(U_D)
        # For Dirichlet sites: F_bottom = -Δ_DD * U_D.
        A_DD = A_DD.tocsr()
        F_bottom = -A_DD.dot(U_D)

        self.F_input = np.concatenate([F_top, F_bottom])
        return self.F_input
    
    def solve(self, A,F, solver="scipy"):
        if solver=="scipy":
            solver_func=scipy_solver
        elif solver=="mumps":
            solver_func=mumps_solver
        elif callable(solver):
            solver_func=solver
        else:
            print("please provide the solver_function")
        """
        Solve the system A_mixed * X = F_input using the provided solver function.

        Parameters:
        - solver_func: a function that accepts (A, F) and returns the solution X.

        Returns:
        - X: the solution vector containing [U_N; n_D].
        """
        return solver_func(A, F)
    
    def solve_capacitance(self,**kwargs):
        N_sites_num=len(self.N_indices)
        D_sites_num=len(self.D_indices)
        n_N = np.zeros(N_sites_num)
        U_D = np.zeros(D_sites_num)
        common_indices_N = list(set(self.material_indices["Qsystem"]).intersection(self.N_indices))
        common_indices_D = list(set(self.material_indices["Qsystem"]).intersection(self.D_indices))
        for idx in common_indices_N:
            n_N[idx]=self.sites[idx].charge
        for idx in common_indices_D:
            U_D[idx]= 1.

        sol= self.solve(self.A_mixed,self.assemble_input(n_N,U_D),**kwargs)
        return sol[N_sites_num:]

    def solve_ni(self,**kwargs):
        N_sites_num=len(self.N_indices)
        n_N = np.zeros(N_sites_num)
        common_indices_N = list(set(self.material_indices["Qsystem"]+self.material_indices["dopants"]).intersection(self.N_indices))
        for idx in common_indices_N:
            n_N[idx]=self.sites[idx].charge
        U_D = np.array([self.sites[i].potential for i in self.D_indices])

        sol= self.solve(self.A_mixed,self.assemble_input(n_N,U_D),**kwargs)
        return sol[N_sites_num:]


    def paa_update_density(self, delta_U, current_density=None, capacitance=None):
        """
        Update the density at each site using the Poisson Adiabatic Approximation (PAA).
        
        The approximation is:
            n_i(new) = n_i(old) + C_i * delta_U_i
        where C_i is the local capacitance at site i and delta_U_i is the change in potential.
        
        Parameters:
        - delta_U: dictionary (or array) mapping site IDs to the change in potential at each site.
        - current_density: (optional) dictionary (or array) of current densities. If not provided,
                            it is assumed that each site has an attribute 'density'.
        - capacitance: (optional) dictionary (or array) of local capacitance values. If not provided,
                        it is assumed that each site has an attribute 'local_capacitance'.
                        
        Returns:
        - updated_density: dictionary mapping site IDs to the updated density.
        """
        updated_density = {}
        
        for i, site in self.sites.items():
            # Get the current density: either from the provided dictionary/array or from the site attribute.
            n_old = current_density[i] if current_density is not None else site.density
            # Get the local capacitance for this site.
            C_i = capacitance[i] if capacitance is not None else site.local_capacitance
            # Get the potential change at this site.
            dU = delta_U[i]
            
            # Apply the PAA formula:
            updated_density[i] = n_old + C_i * dU
            
            # Optionally update the site's density attribute.
            site.density = updated_density[i]
        
        return updated_density
    


def mumps_solver(A, F):
    """
    Solve the sparse system A * X = F using MUMPS via the pymumps package.
    
    Parameters:
    - A: a scipy.sparse matrix in CSR (or converted to CSC) format.
    - F: the right-hand side vector.
    
    Returns:
    - X: the solution vector.
    """
    # Convert the matrix to CSC format, which is generally expected by MUMPS.
    
    # Create a MUMPS context for a real unsymmetric system.
    ctx = mumps.Context()
    if hasattr(ctx, "set_silent"):
        ctx.set_silent(True)
    ctx.set_centralized_system(A, rhs=F)
    
    # Job 6 corresponds to analysis, factorization, and solve.
    ctx.run(job=6)
    
    # Retrieve the solution.
    X = ctx.get_solution()
    
    # Clean up the MUMPS context.
    ctx.destroy()
    
    return X


def scipy_solver(A, F):
    """
    Solve the sparse system A * X = F using SciPy's spsolve.

    Parameters:
    - A: sparse matrix in CSR or CSC format.
    - F: right-hand side vector.

    Returns:
    - X: solution vector.
    """
    X = spsolve(A, F)
    return X