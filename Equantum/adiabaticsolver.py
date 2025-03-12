def paa_update_density(fsc, delta_U, current_density=None, capacitance=None):
        """
        Update the density at each site using the Poisson Adiabatic Approximation (PAA).
        
        The approximation is:
            n_i(new) = n_i(old) + C_i * delta_U_i
        where C_i is the local capacitance at site i and delta_U_i is the change in potential.
        
        Parameters:
        - delta_U: dictionary (or array) mapping site IDs to the change in potential at each site.
        - current_density: (optional) dictionary (or array) of current densities. If not provided,
                            it is assumed that each site has an attribute 'dsensity'.
        - capacitance: (optional) dictionary (or array) of local capacitance values. If not provided,
                        it is assumed that each site has an attribute 'local_capacitance'.
                        
        Returns:
        - updated_density: dictionary mapping site IDs to the updated density.
        """
        updated_density = {}
        
        for i, site in fsc.sites.items():
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