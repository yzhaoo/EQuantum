import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import qsystem as qs
from geometry import Geometry3D
from sites import Site
import def_system_func as systemfunc

# Assume Geometry3D and Site are defined elsewhere and imported.
# For example:
# from geometry3d_module import Geometry3D
# from site_module import Site

class System:
    def __init__(self, geometry_params, config_file=None,ifqsystem=False,quantum_builder="kwant"):
        """
        Initialize the System by building the 3D geometry and assigning site properties.

        Parameters:
          - geometry_params: dict with parameters for Geometry3D, for example:
              {
                "lattice_type": square_lattice,   # or honeycomb_lattice, etc.
                "box_size": ((xmin, xmax), (ymin, ymax), (zmin, zmax)),
                "sampling_density_function": sampling_function,
                "quantum_center": (x0, y0, z0)     # optional, defaults to (0,0,0)
              }
          - config_file: path to a configuration file (e.g., JSON) exported from a 3D builder
                        that defines regions and their physical properties.
        """
        # Initialize Geometry3D
        self.geometry_params = geometry_params
        self.geometry = None
        self.build_geometry()
        # Discretize the simulation box
        self.geometry.discretize()
        print("Generated", len(self.geometry.points), "points in 3D.")
        # Optionally compute the Voronoi tessellation if needed later
        self.geometry.compute_voronoi()
        print("Voronoi cells have been created.")
        
        # Build Site objects from the geometry points.
        # Here we simply create a Site for each point with default values.
        self.sites = systemfunc.create_sites_from_geometry_3d(self.geometry)
        self.num_sites=len(self.sites)
        
        # Load configuration file defining regions and material properties.
        self.config_data = self.load_config(config_file)
        
        # Assign physical properties (material, charge, potential, dielectric, BCtype) to the sites.
        self.assign_properties()
        #initialze the material indices
        self.material_indices = {}  # Dictionary to hold indices by material type.
        self.initialize_material_indices()
        self.Qsites=self.material_indices['Qsystem']
        #initialize the site type
        self.N_indices = [i for i, site in self.sites.items() if site.BCtype == "n"]
        self.D_indices = [i for i, site in self.sites.items() if site.BCtype == "d"]

        #initialize quantum system
        self.qsystem=None
        if ifqsystem:
            self.build_qsystem(quantum_builder)
            print("Quantum system is generated using "+quantum_builder+".")
        print("EQsystem is successfully initialized.")
    def build_geometry(self):
        self.geometry= Geometry3D(lattice_type=self.geometry_params["lattice_type"],
                                   box_size=self.geometry_params["box_size"],
                                   sampling_density_function=self.geometry_params["sampling_density_function"],
                                   quantum_center=self.geometry_params.get("quantum_center", (0.0, 0.0, 0.0)))

    def load_config(self, config_file):
        """
        Load the configuration file (assumed to be in JSON format) that defines regions.

        Parameters:
          - config_file: string, path to the configuration file.
        
        Returns:
          - config_data: the parsed JSON data.
        """
        if config_file==None:
            print("please provide configuration of the setup.")
            return None
        else:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return config_data

    def assign_properties(self):
        """
        Assign physical properties to each site based on the configuration regions.
        
        If a site's z coordinate is approximately zero (within a tolerance),
        the function first checks if it falls within the Qsystem region.
        If so, it assigns the Qsystem properties and skips further region checks.
        Otherwise, it checks the remaining regions.
        """
        regions = self.config_data.get("regions", [])
        
        for site in self.sites.values():
            x, y, z = site.coordinates
            # First, if z is approximately 0, check Qsystem region.
            if z==0.:
                qsystem_found = False
                for region in regions:
                    if region.get("name", "").lower() == "qsystem":
                        bbox = region.get("bbox", {})
                        xmin = bbox.get("xmin", -np.inf)
                        xmax = bbox.get("xmax", np.inf)
                        ymin = bbox.get("ymin", -np.inf)
                        ymax = bbox.get("ymax", np.inf)
                        zmin = bbox.get("zmin", -np.inf)
                        zmax = bbox.get("zmax", np.inf)
                        if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
                            # Assign Qsystem properties.
                            site.material = region.get("name", site.material)
                            site.charge = region.get("charge", site.charge)
                            site.potential = region.get("potential", site.potential)
                            site.dielectric_constant = region.get("dielectric_constant", site.dielectric_constant)
                            site.BCtype = region.get("BCtype", site.BCtype)
                            qsystem_found = True
                            break
                # If the site at z=0 is in the Qsystem region, no need to check other regions.
                if qsystem_found:
                    continue
            
            # For all sites (including those with z != 0 or z==0 that weren't in Qsystem),
            # check each region (if a site falls into multiple regions, you may decide to break after the first match).
            for region in regions:
                bbox = region.get("bbox", {})
                xmin = bbox.get("xmin", -np.inf)
                xmax = bbox.get("xmax", np.inf)
                ymin = bbox.get("ymin", -np.inf)
                ymax = bbox.get("ymax", np.inf)
                zmin = bbox.get("zmin", -np.inf)
                zmax = bbox.get("zmax", np.inf)
                if (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax):
                    site.material = region.get("name", site.material)
                    site.charge = region.get("charge", site.charge)
                    site.potential = region.get("potential", site.potential)
                    site.dielectric_constant = region.get("dielectric_constant", site.dielectric_constant)
                    site.BCtype = region.get("BCtype", site.BCtype)
                    break

    def initialize_material_indices(self):
        """
        Initialize indices for sites based on the material attribute.
        
        This method populates self.material_indices as a dictionary mapping material types
        (e.g., "gate", "dopants", "dielectric", "Qsystem") to a list of site indices.
        
        Returns:
        - material_indices: dict, keys are material names and values are lists of site indices.
        """
        # You can predefine the expected material types if desired.
        expected_materials = ["gate", "dopants", "dielectric", "Qsystem"]
        self.material_indices = {m: [] for m in expected_materials}

        for i, site in self.sites.items():
            mat = site.material if hasattr(site, "material") else None
            if mat is None:
                # Optionally, handle sites with no material attribute.
                continue
            if mat not in self.material_indices:
                self.material_indices[mat] = []
            self.material_indices[mat].append(i)
        return self.material_indices
    
    def build_qsystem(self,quantum_builder):
        self.qsystem=qs.build_system(self.Qsites,builder=quantum_builder)

    def plot_geometry(self, prop=None):
        """
        Plot the discretized sites in 3D space.
        
        If 'prop' is None, sites are colored according to their material (discrete colors).
        Otherwise, 'prop' is expected to be a property name (e.g., "charge") and sites
        will be colored using a continuous colormap based on that property's value.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if prop is None:
            # Group sites by material.
            color_map = {'dielectric':(135/255,213/255,216/255,0.1),
                         'dopants':(225/255,151/255,206/255,0.6),
                         'gate':(255/255,216/255,14/255,1),
                         'Qsystem':(44/255,94/255,84/255,0.8),
                         'vacuum':(1,1,1,0)}
            
            for mat in list(self.material_indices.keys()):
                coords = np.array([self.sites[idx].coordinates for idx in self.material_indices[mat]])
                if len(coords)>0:
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            color=color_map[mat], label=mat, s=10)
        else:
            # Color sites based on a continuous property, e.g., "charge".
            prop_values = []
            coords = []
            for site in self.sites.values():
                # Retrieve the value of the property.
                value = getattr(site, prop, None)
                if value is not None:
                    prop_values.append(value)
                    coords.append(site.coordinates)
            
            if len(coords) == 0:
                raise ValueError(f"No sites have the property '{prop}'.")
            
            coords = np.array(coords)
            prop_values = np.array(prop_values)
            
            # Create a normalization and a ScalarMappable for the colormap.
            norm = mcolors.Normalize(vmin=np.min(prop_values), vmax=np.max(prop_values))
            cmap = cm.viridis
            sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            c=prop_values, cmap=cmap, s=20)
            # Add a colorbar to indicate the property values.
            cbar = fig.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label(prop)
        box_size=self.geometry_params['box_size']
        ax.set_box_aspect((box_size[0][1]-box_size[0][0], box_size[1][1]-box_size[1][0], box_size[2][1]-box_size[2][0]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("System Sites")
        ax.legend()
        plt.show()

