import kwant
import numpy as np


def build_system(syst,builder="kwant",**kwarg):
    if builder=="kwant":
        return kwant_builder(syst,**kwarg)
    






def kwant_builder(syst):
    #define the shape function, cut the system inside the boundary box of the Qsystem
    def qshape(box):
        def ifinshape(pos):
            x,y=pos
            return box[0][0]<=x<=box[0][1] and box[1][0]<=y<=box[1][1]
        return ifinshape
    #define the hopping function, the peierls phase is added as a parameter for later calculation
    def mag_hop(to_site,from_site,phi):
        x=from_site.pos[0]
        return syst.t*np.exp(1j*2*np.pi*phi*x)
    #define the onsite potential, the Ufun is a function that will be defined later
    def onsite_pot(site,Ufunc):
        return Ufunc(site)

    #get the coordinate of the sites belongs to the quantum system
    qcoor=[list(syst.sites.values())[i].coordinates for i in syst.Qsites]
    primi=[(coor[0],coor[1]) for coor in qcoor]

    #create lattice from the coordinates
    qsyst=kwant.Builder()
    lat=kwant.lattice.general([(-1000,0),(0,1000)],primi)

    #cut the finite size sample from the size of Qsystem
    qsitebox=[[min(qcoor[:,0]),max(qcoor[:,0])],[min(qcoor[:,1]),max(qcoor[:,1])]]
    
    #add the initial onsite potential, put zero if no special function is provided
    qsyst[lat.shape(qshape(qsitebox),(0,0))]=onsite_pot

    qsite_idx_map = { site_idx: q_idx for q_idx, site_idx in enumerate(syst.Qsites) }
    #add hopping amplitude, this step takes long
    latsites=lat.sublattices
    for idx in syst.Qsites:
        site = syst.sites[idx]
        qidx = qsite_idx_map[idx]
        for idxn in site.neighbors.keys():
            # Only process if the neighbor is in Qsites.
            if idxn in qsite_idx_map:
                neighbor = syst.sites[idxn]
                # Check if the neighbor's z coordinate is 0.
                if neighbor.coordinates[2] == 0.:
                    qidxn = qsite_idx_map[idxn]
                    xdiff = neighbor.coordinates[0] - site.coordinates[0]
                    qsyst[kwant.builder.HoppingKind((0, 0), latsites[qidx], latsites[qidxn])] = mag_hop

    #(if) add lead here

    return qsyst.finalized()


def kwant_ildos_solver(fsc,murange,**kwarg):
    # use the bulk LDOS (Thomas-Fermi approximation)
    spectrum=kwant.kpm.SpectralDensity(fsc.qsystem,
                                       params=fsc.qparams,
                                       energy_resolution=(murange[1]-murange[0])/5)
    _,densities=spectrum()
    return densities(murange)


