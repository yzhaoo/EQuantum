import kwant
import numpy as np


def build_system(syst,builder="kwant",**kwarg):
    if builder=="kwant":
        return kwant_builder(syst,**kwarg)
    






def kwant_builder(syst,ini_onsite=None,phi=0.):
    #get the coordinate of the sites belongs to the quantum system
    qcoor=[list(syst.sites.values())[i].coordinates for i in syst.Qsites]
    primi=[(coor[0],coor[1]) for coor in qcoor]

    #create lattice from the coordinates
    qsyst=kwant.Builder()
    lat=kwant.lattice.general([(-1000,0),(0,1000)],primi)

    #cut the finite size sample from the size of Qsystem
    qsitebox=[[min(qcoor[:,0]),max(qcoor[:,0])],[min(qcoor[:,1]),max(qcoor[:,1])]]
    def qshape(box):
        def ifinshape(pos):
            x,y=pos
            return box[0][0]<=x<=box[0][1] and box[1][0]<=y<=box[1][1]
        return ifinshape
    #add the initial onsite potential, put zero if no special function is provided
    if ini_onsite==None:
        ini_onsite=0
    qsyst[lat.shape(qshape(qsitebox),(0,0))]=ini_onsite

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
                    qsyst[kwant.builder.HoppingKind((0, 0), latsites[qidx], latsites[qidxn])] = np.exp(1j * 2 * np.pi * xdiff*phi)

    #(if) add lead here

    return qsyst.finalized()