import kwant
from quantum_solvers import kwant_solver as ksolver
from quantum_solvers.default_solver import QuantumSystem
import numpy as np

def build_system(syst,builder="kwant",**kwarg):
    if builder=="kwant":
        qsystem=kwant_solver.kwant_builder(syst,**kwarg)
    elif builder=="default":
        qsystem = QuantumSystem(syst,**kwarg)
    else:
        print("Please provide the quantum builder.")
    return qsystem

def update_params(fsc,params):
    builder=fsc.quantum_builder
    if builder=="kwant":
        fsc.qparams=params
    elif builder=="default":
        fsc.qparams=params
        fsc.qsystem.update_params(params)

def site_map(fsc,syst):
    builder=syst.quantum_builder
    if builder=="kwant":
        ksolver.kwant_site_map_from_Qsites(fsc,syst)
    elif builder=="default":
        fsc.Qsites_map={idx: qidx for idx, qidx in enumerate(fsc.Qsites)}

def update_U(fsc,syst):
    builder=syst.quantum_builder
    if builder=="kwant":
        ksolver.kwant_update_Ufunc(fsc,syst)
    elif builder=="default":
        fsc.qsystem.update_U(fsc)

def update_n(fsc,syst):
    builder=syst.quantum_builder
    if builder=="kwant":
        return ksolver.kwant_density_ED(fsc)
    elif builder=="default":
        return fsc.qsystem.get_dos

def update_ildos(fsc,syst,**kwarg):
    builder=syst.quantum_builder
    if builder=="kwant":
        return ksolver.kwant_ildos_kpm(fsc,**kwarg)
    elif builder=="default":
        cnp= fsc.bandwidth if fsc.lattice_type=="square" else 0*fsc.t
        
        dataall= fsc.qsystem.get_ldos(**kwarg)
        #rescale the filling according to the maximal carrier density
        dataall[:,1,:]*=fsc.max_fill
        #dataall[:,1,:]+=charge_cnp
        #for square lattice, shift the energy, the spectrum start from 0
        dataall[:,0,:]+=cnp
        return dataall
    else:
        print("cannot find the quantum builder.")

def get_n_from_ildos(fsc,edos_data,sample="energy"):
    ##edos_data should be the dos for ee<0 [site,energies,dos]
    nden=np.zeros(len(edos_data))
    if sample == "energy":
        charge_cnp= 0 if fsc.lattice_type=="square" else -fsc.max_fill/2
        filled_idx=[np.where(edos_data[ii,0,:]<=0.)[0][-1] for ii in range(len(edos_data))]
        for ii in range(len(edos_data)):
            nden[ii]=np.sum(edos_data[ii,1,:filled_idx[ii]])
        return nden+charge_cnp


        
