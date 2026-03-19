# self_consistent_solver.py
import poissonsolver as psolver
import qbuilder as qbuilder
import numpy as np
import solvers as solvers
import scipy.constants as sc
import scipy.sparse.linalg as spla

import time

import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython.display import clear_output

class FSC:
    def __init__(self, system, ifinitial=True,qparams=None,convergence_tol=1e-8, max_iter=50,FL_pinning=True,Ncore=1):
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
        self.lattice_type=self.geometry_params['lattice_type']
        self.lat_spacing=system.lat_spacing
        self.unit_cell_area=system.unit_cell_area
        self.unit_cell_area_real=system.unit_cell_area_real
        self.max_fill=system.max_fill
        self.energy_bounds = None
        
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
        self.Qp_in_Q={ii: list(self.Qsites).index(qpidx) for ii,qpidx in enumerate(self.Qprime)}
        
        self.Ncore=Ncore
        self.convergence_tol = convergence_tol
        self.max_iter = max_iter
        self.log = {
            "Qprime_len": [len(self.Qprime)],
            "Ui_maxdiff": [],
            "Ui_l2diff": [],
            "ni_maxdiff": [],
            "ni_l2diff": [],
            "ildos_maxdiff": [],
            "timing_poisson": [],
            "timing_quantum": [],
            "timing_total": [],
        }
        self.snapshots = {}
        self.dashboard_sites = None

        if ifinitial:
            if FL_pinning:
                solvers.Fermi_level_pinning(self)
            #initialize Posisson problem
            self.initial_Poisson()
        if qparams is not None:
            self.update_qparams(system,qparams,ifinitial=False)
            #update the maximal filling for graphene under magnetic field.
            emin, emax = self.get_energy_bounds()
            bandwidth = max(abs(emin), abs(emax))
            self.max_fill = system.max_fill if self.lattice_type=="square" else self.E_to_n(bandwidth, self.qparams['phi'], self.lat_spacing*1e-6)
            #print(self.qparams['phi'],self.lat_spacing,self.bandwidth,self.max_fill,self.E_to_n(self.bandwidth,self.qparams['phi'],self.lat_spacing))
        #initialize Quantum problem
        self.initial_Quantum(system)

    def phi_to_B(self):
        return self.qparams['phi']*sc.h/sc.e/self.unit_cell_area

    def E_to_n(self,ee,phi,a):
        """
        traslate the energy (in the unit of the hopping amplitude) to the carrier density (in the unit of 10^12 cm^-2) according to the LL energy of graphene
        """
        #the carrier density will scale with the chosen lattice spacing. Here the present carrier densiyt is the one after scaling.
        return (4*ee**2/(3*np.pi)+4*phi/np.sqrt(3))/(3* a **2)/1e16
    def get_energy_bounds(self, margin=0.05):
        """
        Estimate spectral bounds of the quantum Hamiltonian.
        """

        H = self.qsystem.get_hamiltonian()

        try:
            emax = spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
            emin = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        except Exception:
            # fallback Gershgorin bound
            abs_rowsum = np.abs(H).sum(axis=1).A.ravel()
            bound = abs_rowsum.max()
            emin, emax = -bound, bound

        width = emax - emin
        pad = margin * width

        bounds = (emin - pad, emax + pad)
        self.energy_bounds = bounds

        return bounds
    
    def initial_Poisson(self):
        """
        initialize the Poisson problem without Quantum system for the given boundary condition.
         
        """
        psolver.calculate_delta(self)
        #initialized Delta_matrix and A_mixed
        self.update_Poisson()
        self.Ci=psolver.solve_capacitance(self)
        print("The poisson problem has been initialized.")

    def update_Poisson(self):
        #solve the initial ND poisson problem and update ni, Ui
        pre_ni = self.ni.copy()
        pre_Ui = self.Ui.copy()

        UnnD = psolver.solve_NDpoisson(self)

        self.ni[self.D_indices] = UnnD[-len(self.D_indices):]
        self.Ui[self.N_indices] = UnnD[:len(self.N_indices)]

        dni = self.ni - pre_ni
        dUi = self.Ui - pre_Ui

        self.log["ni_maxdiff"].append(np.max(np.abs(dni)))
        self.log["ni_l2diff"].append(np.linalg.norm(dni))
        self.log["Ui_maxdiff"].append(np.max(np.abs(dUi)))
        self.log["Ui_l2diff"].append(np.linalg.norm(dUi))
        

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
        self.ildos = qbuilder.update_ildos(
            self,
            system,
            M=256,
            eps=0.05,
            kernel="jackson",
            **kwarg
        )
        self.init_dashboard_sites()
        print("FSC qparams:", self.qparams)
        print("QSystem params:", self.qsystem.qparams)
        print("The quantum problem has been initialized.")


    def update_qparams(self,system,qparams,ifinitial=True):
        qbuilder.update_qparams(self,qparams)
        if ifinitial:
            self.initial_Quantum(system)


    def update_Quantum(self, system, **kwarg):
        pre_ildos = self.ildos.copy()
        qbuilder.update_U(self, system)

        new_ildos = qbuilder.update_ildos(self, system, **kwarg)
        q_indices = list(self.Qp_in_Q.values())

        for dest, src in zip(q_indices, new_ildos):
            self.ildos[dest] = src

        curr_ildos = np.asarray(self.ildos, dtype=float)
        prev_ildos = np.asarray(pre_ildos, dtype=float)

        dildos = curr_ildos - prev_ildos
        self.log["ildos_maxdiff"].append(np.max(np.abs(dildos)))



    def local_solver(self):
        dUdn=solvers.local_solver(self)
        print(np.mean(dUdn[0]),np.mean(dUdn[1]))
        self.Ui[self.Qprime]+=dUdn[0]
        self.ni[self.Qprime]+=dUdn[1]

    def update_BC(self,syst,name,prop,value,ifinitial=False,FL_pinning=True):
        for site in list(self.sites.values()):
            if site.material==name:
                setattr(site, prop, value)
        self.ni=np.array([site.charge for i, site in self.sites.items()])
        self.Ui=np.array([site.potential for i, site in self.sites.items()])
        if ifinitial:
            if FL_pinning:
                solvers.Fermi_level_pinning(self)
            self.initial_Poisson()
            #initialize Quantum problem


    def update_Qprime(self, tol=1e-7):
        Qprime_old = self.Qprime.copy()
        Qprime_new = solvers.update_Qprime(self, tol)
        self.Qprime = Qprime_new

        if len(Qprime_new) != len(Qprime_old):
            psolver.calculate_delta(self)
            self.update_Poisson()
            self.Ci = psolver.solve_capacitance(self)

        self.Qp_in_Q = {ii: list(self.Qsites).index(qpidx) for ii, qpidx in enumerate(self.Qprime)}
        

    def solve(
        self,
        system,
        save=None,
        dashboard_every=5,
        snapshot_every=5,
        max_total_iter=30,
        show_dashboard=False,
        **kwarg
    ):
        """
        Run the self-consistent iteration loop until convergence or max iterations.

        Parameters
        ----------
        system : object
            Underlying system object.
        save : str or None
            Optional filename for saving history.
        dashboard_every : int
            Plot dashboard every N iterations if show_dashboard=True.
        snapshot_every : int
            Save internal snapshots every N iterations.
        max_total_iter : int
            Hard cap on total FSC iterations.
        show_dashboard : bool
            Whether to plot dashboard during solve.
        **kwarg :
            Passed to update_Quantum / qbuilder.update_ildos
        """
        import time

        # initialize active region once
        self.update_Qprime()

        iter_num = [0, 0, 0]   # [Qprime updates, Poisson updates, Quantum updates]

        if save is not None:
            Uis = []
            nis = []
            ildoss = []
            cut_idx = [
                qidx for qidx, idx in enumerate(self.Qsites)
                if np.abs(self.sites[idx].coordinates[0]) < 0.005
                and self.sites[idx].material == 'Qsystem'
            ]
            ildoss = self.ildos[cut_idx]

        while True:
            it = sum(iter_num)
            t_iter0 = time.perf_counter()

            print("The iteration has been conducted for", iter_num, "times.")
            if hasattr(self, "print_iteration_summary"):
                self.print_iteration_summary(it)

            if snapshot_every is not None and snapshot_every > 0:
                if it % snapshot_every == 0:
                    if hasattr(self, "save_snapshot"):
                        self.save_snapshot(it)

            if show_dashboard and dashboard_every is not None and dashboard_every > 0:
                if it % dashboard_every == 0:
                    if hasattr(self, "plot_dashboard"):
                        clear_output(wait=True)
                        self.plot_dashboard(it)

            if save is not None:
                Uis.append(self.Ui.copy())
                nis.append(self.ni.copy())
                self.save_Uini(Uis, nis, ildoss, filename=save)

            # ---------------------------------
            # Step 1: local electrostatic update
            # ---------------------------------
            self.local_solver()

            # ---------------------------------
            # Step 2: update Qprime partition
            # ---------------------------------
            qprime_before = len(self.Qprime)
            self.update_Qprime()
            qprime_after = len(self.Qprime)
            self.log["Qprime_len"].append(qprime_after)

            if qprime_after != qprime_before:
                psolver.calculate_delta(self)

                t0 = time.perf_counter()
                self.Ci = psolver.solve_capacitance(self)
                self.log["timing_poisson"].append(time.perf_counter() - t0)

                iter_num[0] += 1
                self.log["timing_total"].append(time.perf_counter() - t_iter0)

                if sum(iter_num) >= max_total_iter:
                    print("Reached maximum iteration count.")
                    break
                continue

            # ---------------------------------
            # Step 3: Poisson update if needed
            # ---------------------------------
            need_poisson = (
                len(self.log["ni_maxdiff"]) == 0
                or self.log["ni_maxdiff"][-1] > self.convergence_tol
            )

            if need_poisson:
                t0 = time.perf_counter()
                self.update_Poisson()
                self.log["timing_poisson"].append(time.perf_counter() - t0)

                iter_num[1] += 1
                self.log["timing_total"].append(time.perf_counter() - t_iter0)

                if sum(iter_num) >= max_total_iter:
                    print("Reached maximum iteration count.")
                    break
                continue

            # one more Poisson relaxation before quantum step
            t0 = time.perf_counter()
            self.update_Poisson()
            self.log["timing_poisson"].append(time.perf_counter() - t0)

            # ---------------------------------
            # Step 4: Quantum update
            # ---------------------------------
            if sum(iter_num) < max_total_iter:
                t0 = time.perf_counter()
                self.update_Quantum(
                    system,
                    approx="kmeanssample",
                    Ncore=self.Ncore,
                    num_sample=int(5 * self.Ncore),
                    **kwarg
                )
                self.log["timing_quantum"].append(time.perf_counter() - t0)

                if save is not None:
                    ildoss = self.ildos[cut_idx]

                iter_num[2] += 1
                self.log["timing_total"].append(time.perf_counter() - t_iter0)
                continue
            else:
                print("The FSC has been solved.")
                self.log["timing_total"].append(time.perf_counter() - t_iter0)
                break
    # def solve(self,system,save=None,**kwarg):
    #     """
    #     Run the self-consistent iteration loop until convergence or until max_iter is reached.
    #     The loop structure follows Fig.8 of the paper:
    #       - Step I: Update the Q/Q' partition (remove depleted regions)
    #       - Step II: Relax the Poisson (PAA) update (update potential)
    #       - Step III: Relax the quantum (QAA) update (update ILDOS/density)
    #     """
    #     #initialize the problem by conducting iteration twice:
    #     initial_loop=0
    #     self.update_Qprime()
    #     # while initial_loop<2:
    #     #     self.local_solver()
    #     #     self.update_Qprime()
    #     #     psolver.calculate_delta(self)
    #     #     self.Ci=psolver.solve_capacitance(self)
    #     #     initial_loop+=1
    #     iter_num=[0,0,0]
    #     # self.update_Poisson()
    #     #self.update_Quantum(system)
    #     if save is not None:
    #         Uis=[]
    #         nis=[]
    #         ildoss=[]
    #         cut_idx=cut_idx=[qidx for qidx,idx in enumerate(self.Qsites) if np.abs(self.sites[idx].coordinates[0])<0.005 and self.sites[idx].material=='Qsystem']
    #         ildoss=self.ildos[cut_idx]
    #     while True:
    #         t_iter0 = time.perf_counter()
    #         it = sum(iter_num)

    #         print("The iteration has been conducted for ", iter_num, "times.")
    #         self.print_iteration_summary(it)

    #         if it % 5 == 0:
    #             self.save_snapshot(it)
    #             self.plot_dashboard(it)


    #         if save is not None:
    #             Uis.append(self.Ui.copy())
    #             nis.append(self.ni.copy())
    #             self.save_Uini(Uis,nis,ildoss,filename=save)
            
    #         self.local_solver()

    #         t0 = time.perf_counter()
    #         self.local_solver()
    #         t_local = time.perf_counter() - t0

    #         self.update_Qprime()
    #         if self.log['Qprime_len'][-1]-self.log['Qprime_len'][-2]!=0:
    #             psolver.calculate_delta(self)
    #             self.Ci=psolver.solve_capacitance(self)
    #             iter_num[0]+=1
    #             continue
    #         else:
    #             pass
            
    #         if self.log["ni_maxdiff"] and self.log["ni_maxdiff"][-1] > self.convergence_tol:
    #             self.update_Poisson()
    #             iter_num[1]+=1
    #             continue
    #         else:
    #             pass 

    #         t0 = time.perf_counter()
    #         self.update_Poisson()
    #         t_poisson = time.perf_counter() - t0
    #         self.log["timing_poisson"].append(t_poisson) 


    #         # if np.abs(self.log['ildos_error'][-1])>self.convergence_tol:
    #         if np.sum(iter_num)< 30:
    #             self.update_Quantum(system,approx="kmeanssample",Ncore=self.Ncore,num_sample=int(5*self.Ncore),**kwarg)
    #             ildoss=self.ildos[cut_idx]
    #             iter_num[2]+=1
    #             continue
    #         else:
    #             print("The FSC has been solved.")
    #             break
            

    def save_Uini(self,Uis,nis,ildoss,filename):
        mdic={}
        mdic['Uis']=Uis
        mdic['nis']=nis
        mdic['ildoss']=ildoss
        
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


    def plot_qsystem(self, prop_values, title="System Sites", **kwarg):
        """
        Plot the discretized sites in 2D quantum system.

        Parameters
        ----------
        prop_values : array-like
            Values defined on Qsites.
        title : str
            Figure title.
        **kwarg :
            Extra kwargs passed to ax.scatter().
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        coords = np.array([site.coordinates for site in self.sites.values()])[self.Qsites]
        prop_values = np.array(prop_values)

        cmap = cm.viridis
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=prop_values,
            cmap=cmap,
            s=20,
            **kwarg
        )

        fig.colorbar(sc, ax=ax, pad=0.1)
        ax.axis("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        plt.show()
    
    def plot_Hamiltonian(self, ax=None):
        """
        Plot the Hamiltonian of the quantum system in 2D.

        Sites are plotted as dots colored by onsite potential.
        Hoppings are plotted as lines colored by Re(H_ij).
        """

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import numpy as np

        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8))

        # Hamiltonian
        H = self.qsystem.get_hamiltonian().tocsr()

        # Qsystem site coordinates
        coords = np.array([self.sites[idx].coordinates for idx in self.Qsites])
        xy = coords[:, :2]

        # onsite potential
        onsite = H.diagonal().real

        # --- plot sites ---
        norm_sites = mcolors.Normalize(vmin=onsite.min(), vmax=onsite.max())
        cmap_sites = cm.viridis

        sc = ax.scatter(
            xy[:,0],
            xy[:,1],
            c=onsite,
            cmap=cmap_sites,
            s=30,
            norm=norm_sites,
            zorder=3
        )

        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Onsite potential")

        # --- extract hoppings ---
        rows, cols = H.nonzero()

        mask = rows < cols   # avoid double plotting
        rows = rows[mask]
        cols = cols[mask]

        hop = H[rows, cols].A1.real

        norm_hop = mcolors.Normalize(vmin=hop.min(), vmax=hop.max())
        cmap_hop = cm.coolwarm

        # --- plot hopping lines ---
        for i, j, t in zip(rows, cols, hop):

            x = [xy[i,0], xy[j,0]]
            y = [xy[i,1], xy[j,1]]

            ax.plot(
                x,
                y,
                color=cmap_hop(norm_hop(t)),
                linewidth=1.0,
                alpha=0.8,
                zorder=1
            )

        sm = cm.ScalarMappable(norm=norm_hop, cmap=cmap_hop)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax, pad=0.08)
        cbar2.set_label("Re(Hopping)")

        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Quantum Hamiltonian")

        plt.show()

########debug log info############        
    def save_snapshot(self, it):
        snap = {
            "Ui": self.Ui.copy(),
            "ni": self.ni.copy(),
            "Qprime": self.Qprime.copy(),
        }

        if self.ildos is not None and self.dashboard_sites is not None:
            snap["ldos_sites"] = {}
            for site in self.dashboard_sites:
                try:
                    snap["ldos_sites"][site] = np.array(self.ildos[site], dtype=object)
                except Exception:
                    pass

        self.snapshots[it] = snap

    def print_iteration_summary(self, iter_num):
        print(f"\n--- Iteration {iter_num} ---")
        print(f"Qprime size   : {len(self.Qprime)}")

        if self.log["Ui_maxdiff"]:
            print(f"max |ΔU|      : {self.log['Ui_maxdiff'][-1]:.3e}")
            print(f"L2(ΔU)        : {self.log['Ui_l2diff'][-1]:.3e}")

        if self.log["ni_maxdiff"]:
            print(f"max |Δn|      : {self.log['ni_maxdiff'][-1]:.3e}")
            print(f"L2(Δn)        : {self.log['ni_l2diff'][-1]:.3e}")

        if self.log["ildos_maxdiff"]:
            print(f"max |ΔILDOS|  : {self.log['ildos_maxdiff'][-1]:.3e}")

    def plot_convergence(self):

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        ax[0].plot(self.log["Ui_maxdiff"])
        ax[0].set_yscale("log")
        ax[0].set_title("max |ΔU|")

        ax[1].plot(self.log["ni_maxdiff"])
        ax[1].set_yscale("log")
        ax[1].set_title("max |Δn|")

        ax[2].plot(self.log["Qprime_len"])
        ax[2].set_title("Qprime size")

        plt.tight_layout()
        plt.show()

    def init_dashboard_sites(self, n_sites=4):
        coords = np.array([self.sites[idx].coordinates[:2] for idx in self.Qsites])

        # pick corners + center approximately
        center = coords.mean(axis=0)
        d_center = np.linalg.norm(coords - center, axis=1)
        center_idx = np.argmin(d_center)

        x_min_idx = np.argmin(coords[:, 0])
        x_max_idx = np.argmax(coords[:, 0])
        y_min_idx = np.argmin(coords[:, 1])
        y_max_idx = np.argmax(coords[:, 1])

        chosen = list(dict.fromkeys([center_idx, x_min_idx, x_max_idx, y_min_idx, y_max_idx]))
        self.dashboard_sites = chosen[:n_sites]

    def plot_dashboard(self, it=None):

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax_conv, ax_U, ax_n, ax_ldos = axes.ravel()

        # ----------------------------
        # Panel 1: convergence curves
        # ----------------------------
        if len(self.log["Ui_maxdiff"]) > 0:
            ax_conv.plot(self.log["Ui_maxdiff"], label="max |ΔU|")
        if len(self.log["ni_maxdiff"]) > 0:
            ax_conv.plot(self.log["ni_maxdiff"], label="max |Δn|")
        if len(self.log["ildos_maxdiff"]) > 0:
            ax_conv.plot(self.log["ildos_maxdiff"], label="max |ΔILDOS|")

        ax_conv.set_yscale("log")
        ax_conv.set_title("Convergence")
        ax_conv.legend()

        # ----------------------------
        # coordinates on Qsites
        # ----------------------------
        coords = np.array([self.sites[idx].coordinates[:2] for idx in self.Qsites])

        # ----------------------------
        # Panel 2: Ui
        # ----------------------------
        Ui_q = self.Ui[self.Qsites]
        sc1 = ax_U.scatter(coords[:, 0], coords[:, 1], c=Ui_q, cmap="coolwarm", s=30)
        fig.colorbar(sc1, ax=ax_U, pad=0.02)
        ax_U.set_aspect("equal")
        ax_U.set_title("Ui on Qsites")

        # ----------------------------
        # Panel 3: ni
        # ----------------------------
        ni_q = self.ni[self.Qsites]
        sc2 = ax_n.scatter(coords[:, 0], coords[:, 1], c=ni_q, cmap="viridis", s=30)
        fig.colorbar(sc2, ax=ax_n, pad=0.02)
        ax_n.set_aspect("equal")
        ax_n.set_title("ni on Qsites")

        # ----------------------------
        # Panel 4: tracked LDOS
        # ----------------------------
        if self.ildos is not None and self.dashboard_sites is not None:
            for site in self.dashboard_sites:
                try:
                    E = np.asarray(self.ildos[site][0], dtype=float)
                    rho = np.asarray(self.ildos[site][1], dtype=float)
                    ax_ldos.plot(E, rho, label=f"site {site}")
                except Exception:
                    pass

        ax_ldos.set_title("Tracked LDOS")
        ax_ldos.set_xlabel("Energy")
        ax_ldos.set_ylabel("LDOS")
        if self.dashboard_sites is not None:
            ax_ldos.legend(fontsize=8)

        if it is not None:
            fig.suptitle(f"FSC dashboard — iteration {it}", fontsize=14)

        plt.tight_layout()
        plt.show()