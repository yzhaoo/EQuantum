# self_consistent_solver.py
import poissonsolver as psolver
import qbuilder as qbuilder
import numpy as np
import solvers as solvers

import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.cm as cm
import matplotlib.colors as mcolors
class FSC:
    def __init__(self, system, ifinitial=True,params=None,convergence_tol=1e-10, max_iter=50):
        """
        Initialize the self-consistent solver.

        Parameters:
          - system: an instance of your System class (which includes Geometry3D, Site objects, etc.)
          - quantum_solver: an instance (or module) that performs the quantum calculation (QAA update)
          - poisson_solver: an instance (or module) that performs the electrostatic/Poisson calculation (PAA update)
          - convergence_tol: tolerance for convergence.
          - max_iter: maximum number of iterations allowed.
        """
        #quantities
        #initialized from System
        self.geometry_params=system.geometry_params
        self.quantum_builder=system.quantum_builder
        self.sites = (system.sites).copy()
        self.num_sites=system.num_sites
        self.material_indices=system.material_indices
        self.ni=np.array([site.charge for i, site in self.sites.items()])
        self.Ui=np.array([site.potential for i, site in self.sites.items()])
        self.N_indices = (system.N_indices).copy()
        self.D_indices = (system.D_indices).copy()
        self.t=system.t
        self.lat_spacing=system.lat_spacing
        self.unit_cell_area=system.unit_cell_area
        self.unit_cell_area_real=system.unit_cell_area_real
        self.max_fill=system.max_fill
        #initialize with Poisson solver parameters
        self.Ci=None
        self.A_mixed=None
        self.F_input=None
        self.Delta_matrix= None

        #initialize with quantum solver parameters
        self.qparams={}
        self.ildos=None
        self.Qsites=system.Qsites
        self.Qprime=system.Qsites.copy()
        self.qsystem=system.qsystem
        self.Qsites_map={}
        
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.log={'ni_error':[1],
        'Qprime_len':[],
        'ildos_error':[1]}

        if ifinitial:
            #initialize Posisson problem
            self.initial_Poisson()
        if params is not None:
            self.update_qparams(system,params,ifinitial=False)
        #initialize Quantum problem
        self.initial_Quantum(system)
    
    def initial_Poisson(self):
        """
        initialize the Poisson problem without Quantum system for the given boundary condition.
         
        """
        #initialized Delta_matrix and A_mixed
        self.update_Poisson()
        print("The poisson problem has been initialized.")

    def update_Poisson(self):
        psolver.calculate_delta(self)
        #solve the initial ND poisson problem and update ni, Ui
        pre_ni=self.ni.copy()
        UnnD=psolver.solve_NDpoisson(self)
        self.ni[self.D_indices]=UnnD[-len(self.D_indices):]
        self.log['ni_error'].append(np.mean(pre_ni-self.ni))
        self.Ui[self.N_indices]=UnnD[:len(self.N_indices)]
        #initialized Ci
        self.Ci=psolver.solve_capacitance(self)
        

    def initial_Quantum(self,system,**kwarg):
        """
        initialize the Quantum problem without the external electristatic field, yield initial ILDOS

        """
        #initialize the site map between Qsysetm and kwant system
        qbuilder.site_map(self,system)
        #initialize the potential function Ufunc
        qbuilder.update_U(self,system)
        #initialize at the half-filling (since assume U=0 onsite)
        #self.ni[self.Qsites]+=0.5*np.ones(len(self.Qsites))
        #calculate the initial ildos
        self.ildos=qbuilder.update_ildos(self,system,delta=self.t/20,w=np.linspace(-3.9*self.t,3.9*self.t,2000),
npol_scale=6,**kwarg)
        print("The quantum problem has been initialized.")
    def update_qparams(self,system,params,ifinitial=True):
        qbuilder.update_params(self,params)
        if ifinitial:
            self.initial_Quantum(system)


    def update_Quantum(self,system,**kwarg):
        pre_ildos=self.ildos.copy()
        qbuilder.update_U(self,system)
        self.ildos=qbuilder.update_ildos(self,system,delta=self.t/100,w=np.linspace(-5*self.t,5*self.t,200),**kwarg)
        self.log['ildos_error'].append(np.mean(self.ildos-pre_ildos))
        self.ni[self.Qsites]=qbuilder.get_n_from_ildos(self,self.ildos)



    def local_solver(self):
        dUdn=solvers.local_solver(self)
        print(np.mean(dUdn[0]),np.mean(dUdn[1]))
        self.Ui[self.Qprime]+=dUdn[0]
        self.ni[self.Qprime]+=dUdn[1]

    def update_BC(self,syst,name,prop,value,ifinitial=False):
        for site in list(self.sites.values()):
            if site.material==name:
                setattr(site, prop, value)
        if ifinitial:
            self.initial_Poisson()
            #initialize Quantum problem


    def update_Qprime(self,tol=1e-7):
        Qprime_new=solvers.update_Qprime(self,tol)
        self.log['Qprime_len'].append(len(Qprime_new))
        #print(self.log['Qprime_len'])
        #print(len(Qprime_new))
        self.Qprime=Qprime_new


    def solve(self,system,save=None):
        """
        Run the self-consistent iteration loop until convergence or until max_iter is reached.
        The loop structure follows Fig.8 of the paper:
          - Step I: Update the Q/Q' partition (remove depleted regions)
          - Step II: Relax the Poisson (PAA) update (update potential)
          - Step III: Relax the quantum (QAA) update (update ILDOS/density)
        """
        #initialize the problem by conducting iteration twice:
        initial_loop=0
        self.update_Qprime()
        while initial_loop<2:
            self.local_solver()
            self.update_Qprime()
            psolver.calculate_delta(self)
            self.Ci=psolver.solve_capacitance(self)
            initial_loop+=1
        iter_num=[0,0,0]
        self.update_Poisson()
        #self.update_Quantum(system)
        if save is not None:
            Uis=[]
            nis=[]
        while True:
            print("The iteration has been conducted for ", iter_num,"times.")
            print(self.log)
            if save is not None:
                Uis.append(self.Ui)
                nis.append(self.ni)
                self.save_Uini(Uis,nis,filename=save)
            
            self.local_solver()
            self.update_Qprime()
            if self.log['Qprime_len'][-1]-self.log['Qprime_len'][-2]!=0:
                psolver.calculate_delta(self)
                self.Ci=psolver.solve_capacitance(self)
                iter_num[0]+=1
                continue
            else:
                pass
            
            if np.abs(self.log['ni_error'][-1])>self.convergence_tol:
                self.update_Poisson()
                iter_num[1]+=1
                continue
            else:
                pass 
            self.update_Poisson()   

            break
            # if np.abs(self.log['ildos_error'][-1])>self.convergence_tol:
            #     self.update_Quantum(system)
            #     iter_num[2]+=1
            #     continue
            # else:
            #     print("The FSC has been solved.")
            #     break
            

    def save_Uini(self,Uis,nis,filename):
        mdic={}
        mdic['Uis']=Uis
        mdic['nis']=nis
        from scipy.io import savemat
        mat_fname=filename
        savemat(mat_fname,mdic)


    def plot_full(self, prop_values,**kwarg):
        """
        Plot the discretized sites in 3D space.
        
        If 'prop' is None, sites are colored according to their material (discrete colors).
        Otherwise, 'prop' is expected to be a property name (e.g., "charge") and sites
        will be colored using a continuous colormap based on that property's value.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Color sites based on a continuous property, e.g., "charge".
        coords = [site.coordinates for site in self.sites.values()]

        
        coords = np.array(coords)
        prop_values = np.array(prop_values)
        
        # Create a normalization and a ScalarMappable for the colormap.
        #norm = mcolors.Normalize(vmin=np.min(prop_values), vmax=np.max(prop_values))
        cmap = cm.viridis
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=prop_values, cmap=cmap, s=20,vmin=np.min(prop_values), vmax=np.max(prop_values),**kwarg)
        # Add a colorbar to indicate the property values.
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        #cbar.set_label(prop_values)
        box_size=self.geometry_params['box_size']
        ax.set_box_aspect((box_size[0][1]-box_size[0][0], box_size[1][1]-box_size[1][0], box_size[2][1]-box_size[2][0]))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("System Sites")
        ax.legend()
        plt.show()


    def plot_qsystem(self,prop_values,**kwarg):
        """
        Plot the discretized sites in 2d quantum system
        
        If 'prop' is None, sites are colored according to their material (discrete colors).
        Otherwise, 'prop' is expected to be a property name (e.g., "charge") and sites
        will be colored using a continuous colormap based on that property's value.
        """
        fig, ax=plt.subplots(figsize=(10,8))

        # Color sites based on a continuous property, e.g., "charge".
        coords = np.array([site.coordinates for site in self.sites.values()])[self.Qsites]
        prop_values = np.array(prop_values)
        
        # Create a normalization and a ScalarMappable for the colormap.
        #norm = mcolors.Normalize(vmin=np.min(prop_values), vmax=np.max(prop_values))
        cmap = cm.viridis
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=prop_values, cmap=cmap, s=20,**kwarg)
        # Add a colorbar to indicate the property values.
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        #cbar.set_label(prop_values)
        #box_size=self.geometry_params['box_size']
        #ax.set_box_aspect((box_size[0][1]-box_size[0][0], box_size[1][1]-box_size[1][0], box_size[2][1]-box_size[2][0]))
        ax.axis('equal')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("System Sites")
        ax.legend()
        plt.show()