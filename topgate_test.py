from sites import Site
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from fsc import FSC
import scipy.linalg as sl
import importlib
import kwant
from EQsystem import System
def density_function(r):
        spacing0 = 0.023 # spacing at r=0
        k = 0.6    # spacing increases by 0.05 per unit distance
        return spacing0 + k * r

    # Define a 3D simulation box: ((xmin, xmax), (ymin, ymax), (zmin, zmax))

testparams={"lattice_type": "square",   # or honeycomb_lattice, etc.
"box_size": ((-1.7, 1.7), (-3.2, 3.2), (-0.5, 0.5)),
"sampling_density_function": density_function,
"quantum_center": (0,0,0)     # optional, defaults to (0,0,0)
              }

syst=System(testparams,config_file="setup/updated_sites.json",ifqsystem=True,t=1,quantum_builder="default")
fsc=FSC(syst,ifinitial=True,params={'Ufunc': lambda x:0,'phi':2.4},convergence_tol=1e-6)
fsc.solve(syst,save=True)