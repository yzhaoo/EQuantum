import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

def local_solver_i(idx,ildos,Ci,ni,Ui,limits,cnp):
    #interpolate the ildos to be a conitnuous function
    x_dis=ildos[0]-Ui
    y_dis=ildos[1]
    ildos_dis=[np.sum(y_dis[:idx]) for idx in range(len(x_dis))]
    ildos_dis=ildos_dis-cnp
    ildos_iterp=interp1d(x_dis,ildos_dis,kind='linear',fill_value='exptrapolate')
    def dn_for_Ci(dU):
        return dU*Ci+ni
    def diff(dU):
        return np.abs(dn_for_Ci(dU)-ildos_iterp(dU))
    try:
        result = minimize_scalar(diff, bounds=limits, method='bounded')
    except ValueError:
        print(idx)
    dUsol=result.x
    dnsol=dn_for_Ci(dUsol)-ni
    return dUsol,dnsol
        
    
    return dUsol,dnsol

def local_solver(fsc):
    dUs=np.zeros(len(fsc.Qprime))
    dns=np.zeros(len(fsc.Qprime))
    
    for ii in range(len(fsc.Qprime)):
        Uii=fsc.Ui[fsc.Qprime][ii]
        if fsc.lattice_type=="square":
            elimit=(0-Uii,2*fsc.bandwidth-Uii)
            charge_cnp= 0 
        else:
            elimit=(-Uii-0*fsc.bandwidth,2*fsc.bandwidth-Uii)
            charge_cnp=0*fsc.max_fill/2
        dU,dn=local_solver_i(ii,fsc.ildos[fsc.Qp_in_Q[ii]],fsc.Ci[ii],fsc.ni[fsc.Qprime][ii],Uii,elimit,charge_cnp)
        dUs[ii]=dU
        dns[ii]=dn
    return [dUs,dns]

def update_Qprime(fsc,tol=0):
    Qprime=fsc.Qprime.copy()
    #delete the idx from self.Qprime if the local ni==0
    remove_idx=[]
    for idx,qsite in enumerate(fsc.Qprime):
        if fsc.ni[qsite]<tol or fsc.ni[qsite]>0.95*fsc.max_fill: #or fsc.Ci[idx]<-2:
            remove_idx.append(idx)
    if remove_idx !=[]:
        fsc.N_indices=np.array(list(set(np.append(fsc.N_indices, np.array(Qprime)[remove_idx]))))
        fsc.D_indices=np.array(list(set(range(fsc.num_sites))-set(fsc.N_indices)))
    return np.delete(Qprime, remove_idx)

def Fermi_level_pinning(fsc):
    #remove the sites connecting to the contact from the fsc system
    Qprime=fsc.Qprime.copy()
    #delete the idx from self.Qprime if the local ni==0
    remove_idx=[]
    for qidx,idx in enumerate(fsc.Qprime):
        for neighbor in fsc.sites[idx].neighbors:
            if fsc.sites[neighbor].material=='gate':
                remove_idx.append(qidx)
                fsc.ni[idx]=fsc.max_fill
                # if fsc.sites[neighbor].potential > fsc.bandwidth:
                #     fsc.sites[idx].potential=fsc.bandwidth
                #     #print("the potential of the sites under the contact has been fixed to the bandwidth.")
                # else:
                #     fsc.sites[idx].potential=fsc.sites[neighbor].potential
                #     #print("the potential of the sites under the contact has been fixed to the gate potential.")
                break
    if remove_idx !=[]:
        fsc.N_indices=np.array(list(set(np.append(fsc.N_indices, np.array(Qprime)[remove_idx]))))
        fsc.D_indices=np.array(list(set(range(fsc.num_sites))-set(fsc.N_indices)))
    fsc.Qprime= np.delete(Qprime, remove_idx)
