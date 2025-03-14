import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

def local_solver_i(idx,ildos,Ci,ni,Ui,limits):
    #interpolate the ildos to be a conitnuous function
    x_dis=ildos[0]-Ui
    y_dis=ildos[1]
    ildos_dis=[np.sum(y_dis[:idx]) for idx in range(len(x_dis))]
    ildos_iterp=interp1d(x_dis,ildos_dis,kind='linear',fill_value='exptrapolate')
    def dn_for_Ci(dU):
        return -dU*Ci+ni
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
            elimit=(0-Uii,fsc.bandwidth-Uii)
        else:
            elimit=(-Uii-fsc.bandwidth/2,fsc.bandwidth/2-Uii)
        dU,dn=local_solver_i(ii,fsc.ildos[ii],fsc.Ci[ii],fsc.ni[fsc.Qprime][ii],Uii,elimit)
        dUs[ii]=dU
        dns[ii]=dn
    return [dUs,dns]

def update_Qprime(fsc,tol=0):
    Qprime=fsc.Qprime.copy()
    #delete the idx from self.Qprime if the local ni==0
    for idx,qsite in enumerate(fsc.Qprime):
        if fsc.ni[qsite]<tol:
            np.delete(fsc.Qprime,idx)
    return Qprime
