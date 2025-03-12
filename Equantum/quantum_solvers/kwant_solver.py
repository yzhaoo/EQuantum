import kwant
import numpy as np
from tqdm import tqdm
import scipy.linalg as sl
from scipy.spatial import KDTree


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

    #get the coordinate of the sites belongs to the quantum system, keep 3d coords for vertical lead
    qcoor=np.array([list(syst.sites.values())[i].coordinates for i in syst.Qsites])
    primi=[(coor[0],coor[1]) for coor in qcoor]

    #create lattice from the coordinates
    qsyst=kwant.Builder()
    base_spacing_func = syst.geometry_params['sampling_density_function']
    lat=kwant.lattice.general([(-100,0,),(0,100)],primi,norbs=1)

    #cut the finite size sample from the size of Qsystem
    qsitebox=[[min(qcoor[:,0]),max(qcoor[:,0])],[min(qcoor[:,1]),max(qcoor[:,1])]]
    
    #add the initial onsite potential, put zero if no special function is provided
    qsyst[lat.shape(qshape(qsitebox),(0,0))]=onsite_pot

    qsite_idx_map = { site_idx: q_idx for q_idx, site_idx in enumerate(syst.Qsites) }
    #add hopping amplitude, this step takes long
    latsites=lat.sublattices
    for idx in tqdm(syst.Qsites):
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
# def kwant_lead(fsc,lattice):
#     #generate lead shape functions
#     lead_shape_funcs=[]
#     for lead_i_sites in fsc.Lead_sites:
#         def lead_shape(pos):
#             x,y,z=pos
#             cross=[np.cross(lsite, np.array([x,y,z])) for lsite in lead_i_sites]
#             if cross.any():
#                 return True
#         lead_shape_funcs.append(lead_shape)

#         #build lead system
#         leadsys=kwant.TranslationalSymmetry((0,0,1))
#         lead=kwant.Builder(leadsys)
#         lead[lattice.shape(lead_shape,lead_i_sites[1])]=1
#         qsite_idx_map = { site_idx: q_idx for q_idx, site_idx in enumerate(syst.Qsites) }
#         #add hopping amplitude, this step takes long
#         latsites=lat.sublattices
#         for idx in tqdm(syst.Qsites):
#             site = syst.sites[idx]
#             qidx = qsite_idx_map[idx]
#             for idxn in site.neighbors.keys():
#                 # Only process if the neighbor is in Qsites.
#                 if idxn in qsite_idx_map:
#                     neighbor = syst.sites[idxn]
#                     # Check if the neighbor's z coordinate is 0.
#                     if neighbor.coordinates[2] == 0.:
#                         qidxn = qsite_idx_map[idxn]
#                         xdiff = neighbor.coordinates[0] - site.coordinates[0]
#                         qsyst[kwant.builder.HoppingKind((0, 0,0), latsites[qidx], latsites[qidxn])] = mag_hop

        

# def kwant_add_hop_lead(latsites,hop_func):
#     for idx in tqdm(syst.Qsites):
#             site = syst.sites[idx]
#             qidx = qsite_idx_map[idx]
#             for idxn in site.neighbors.keys():
#                 # Only process if the neighbor is in Qsites.
#                 if idxn in qsite_idx_map:
#                     neighbor = syst.sites[idxn]
#                     # Check if the neighbor's z coordinate is 0.
#                     if neighbor.coordinates[2] == 0.:
#                         qidxn = qsite_idx_map[idxn]
#                         xdiff = neighbor.coordinates[0] - site.coordinates[0]
#                         qsyst[kwant.builder.HoppingKind((0, 0,0), latsites[qidx], latsites[qidxn])] = mag_hop

            


def kwant_site_map_from_Qsites(fsc,syst):
    #key: index in Qsites, values: index in Kwant sites
    kcoord=np.array([site.pos for site in syst.qsystem.sites])
    qcoord=np.array([list(syst.sites.values())[i].coordinates for i in syst.Qsites])
    # Build a KDTree on the Kwant coordinates.
    tree = KDTree(kcoord)
    # Define a threshold for matching (this depends on your precision; adjust as needed)
    threshold = 1e-8
    # Create a dictionary (or array) to store the mapping: 
    # mapping[i] will be the index in kwant_coords corresponding to input_coords[i].
    mapping = {}
    for i, coord in enumerate(qcoord):
        # Query the KDTree for the nearest neighbor.
        dist, idx = tree.query(coord)
        # Optionally, you can check if the distance is below a threshold.
        if dist < threshold:
            mapping[i] = idx
        else:
            # Even if the distance is larger than expected, you might decide to assign it anyway.
            mapping[i] = idx
    # mapping now gives the index correspondence between input_coords and kwant_coords.
    fsc.Qsites_map=mapping

def kwant_update_Ufunc(fsc,syst):
    ksite_dict={ ksite : k_idx for k_idx, ksite in enumerate(syst.qsystem.sites)}
    kUlist=np.array([fsc.Ui[syst.Qsites[qidx]] for qidx in range(len(fsc.Qsites_map))])
    kUlist=kUlist[list(fsc.Qsites_map.values())]
    def Ufunc(site):
        return kUlist[ksite_dict[site]]
    fsc.qparams['Ufunc']=Ufunc

def kwant_density_ED(fsc):
    k_to_q_map=np.argsort(list(fsc.Qsites_map.values()))
    ham_mat=fsc.qsystem.hamiltonian_submatrix(params=fsc.qparams)
    ew,ev=sl.eigh(ham_mat)
    sort_idx=np.argsort(ew)

    qnden=np.zeros(len(fsc.Qsites_map))

    for eidx,ee in enumerate(ew):
        if ee <0:
            kdos=np.abs(ev[sort_idx[eidx],:])
            qnden[eidx]+=kdos[k_to_q_map]

    return qnden

def kwant_ildos_kpm(fsc,**kwarg):
    # use the bulk LDOS (Thomas-Fermi approximation)
    spectrum=kwant.kpm.SpectralDensity(fsc.qsystem,
                                       params=fsc.qparams,
                                       energy_resolution=0.02,**kwarg)
    return spectrum()





# def kwant_builder_3d(syst):
#     #define the shape function, cut the system inside the boundary box of the Qsystem
#     def qshape(box):
#         def ifinshape(pos):
#             x,y,z=pos
#             return box[0][0]<=x<=box[0][1] and box[1][0]<=y<=box[1][1] and box[2][0]<=z< box[2][1]
#         return ifinshape
#     #define the hopping function, the peierls phase is added as a parameter for later calculation
#     def mag_hop(to_site,from_site,phi):
#         x=from_site.pos[0]
#         return syst.t*np.exp(1j*2*np.pi*phi*x)
#     #define the onsite potential, the Ufun is a function that will be defined later
#     def onsite_pot(site,Ufunc):
#         return Ufunc(site)

#     #get the coordinate of the sites belongs to the quantum system, keep 3d coords for vertical lead
#     qcoor=np.array([list(syst.sites.values())[i].coordinates for i in syst.Qsites])
#     primi=[(coor[0],coor[1],coor[2]) for coor in qcoor]

#     #create lattice from the coordinates
#     qsyst=kwant.Builder()
#     base_spacing_func = syst.geometry_params['sampling_density_function']
#     lat=kwant.lattice.general([(-100,0,0),(0,100,0),(0,0,base_spacing_func(0.0))],primi,norbs=1)

#     #cut the finite size sample from the size of Qsystem
#     qsitebox=[[min(qcoor[:,0]),max(qcoor[:,0])],[min(qcoor[:,1]),max(qcoor[:,1])],[0,base_spacing_func(0.0)]]
    
#     #add the initial onsite potential, put zero if no special function is provided
#     qsyst[lat.shape(qshape(qsitebox),(0,0,0))]=onsite_pot

#     qsite_idx_map = { site_idx: q_idx for q_idx, site_idx in enumerate(syst.Qsites) }
#     #add hopping amplitude, this step takes long
#     latsites=lat.sublattices
#     for idx in tqdm(syst.Qsites):
#         site = syst.sites[idx]
#         qidx = qsite_idx_map[idx]
#         for idxn in site.neighbors.keys():
#             # Only process if the neighbor is in Qsites.
#             if idxn in qsite_idx_map:
#                 neighbor = syst.sites[idxn]
#                 # Check if the neighbor's z coordinate is 0.
#                 if neighbor.coordinates[2] == 0.:
#                     qidxn = qsite_idx_map[idxn]
#                     xdiff = neighbor.coordinates[0] - site.coordinates[0]
#                     qsyst[kwant.builder.HoppingKind((0, 0,0), latsites[qidx], latsites[qidxn])] = mag_hop

#     #(if) add lead here

#     return qsyst.finalized()