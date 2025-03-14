import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm

class QuantumSystem:
    def __init__(self, syst, params={'Ufunc': lambda x: 0,'phi':0.}):
        """
        Initialize the quantum system.
        
        Parameters:
          Qsites: list of site dictionaries.
          params: dictionary of parameters, for example:
              {
                  'Ufunc': <callable that takes a site and returns an onsite energy>,
                  'mag_hop': <callable that takes (site_i, site_j, phi) and returns a hopping amplitude>,
                  'phi': <phase parameter>
              }
        """
        self.Qsites = [syst.sites[idx] for idx in syst.Qsites]
        self.q_to_Q_map={qidx:idx for idx, qidx in enumerate(syst.Qsites)}
        self.all_sites=syst.sites
        self.params = params.copy()  # Store the parameter dictionary
        self.N = len(self.Qsites)
        self.lattice_type=syst.geometry_params['lattice_type']
        self.lat_spacing=syst.lat_spacing
        self.t=syst.t
        self.H = None
        self.build_hamiltonian()
    
    def build_hamiltonian(self):
        Ufunc = self.params['Ufunc']  # This is a function we haven't called yet.
        phi = self.params['phi']
        lat_spacing=self.lat_spacing

        hop_func= mag_hop_square if self.lattice_type=="square" else mag_hop_honeycomb
        
        
        self.H = lil_matrix((self.N, self.N), dtype=np.complex128)
        
        for i, site in enumerate(self.Qsites):
            # Diagonal term: call the onsite function with the site.
            self.H[i, i] = onsite_pot(site, Ufunc) 
            # Off-diagonal: iterate over neighbors
            for j in site.neighbors:
                if np.abs(self.all_sites[j].coordinates[2])<1e-8 and self.all_sites[j].material=='Qsystem':
                    try:
                        j_idx=self.q_to_Q_map[j]
                    except KeyError:
                            print(j)
                    neighbor = self.Qsites[j_idx]
                    coord_j = np.array(neighbor.coordinates)
                    # Only add hopping if neighbor's z coordinate is near 0.
                    hop_val = hop_func(self.t,site, neighbor, phi,lat_spacing)
                    #hop_val_back=hop_func(self.t, neighbor,site, phi,lat_spacing)
                    self.H[i, j_idx] = hop_val
                    #self.H[j_idx, i] = hop_val_back
    
    def update_params(self, new_params):
        """
        Update the parameters (such as Ufunc or phi) and rebuild the Hamiltonian.
        """
        self.params=new_params
        self.build_hamiltonian()

    def update_U(self,fsc):
        def Ufunc(site):
            return -fsc.Ui[site.id]
        fsc.qparams['Ufunc']=Ufunc
        self.update_params(fsc.qparams)
    
    def get_hamiltonian(self):
        return self.H.tocsr()

    def get_dos(self,params=None,i=None,ntries=10,**kwargs):
        """(from qtcipy) Return the DOS, averaging over several vectors. default:10"""
        if params is not None:
            self.update_params(params)
        m=self.H.toarray()
        o = []
        d0 = 0.
        if i is None: i = [ii for ii in range(m.shape[0])]
        for j in range(ntries):
            e,d = get_dos_i(m,i=i,**kwargs)
            d0 = d0 + d
        d0=d0/np.sum(d0)
        return e,d0 #/ntries
    def get_ldos(self,params=None,TFapprox=True,ifpara=False,Ncore=5,**kwargs):
        if TFapprox:
            bulk_dos= self.get_dos(**kwargs)
            
            dataall=[bulk_dos for _ in range(len(self.Qsites))]
        else:
            if ifpara:
                dataall=Parallel(n_jobs=Ncore)(delayed(self.get_dos)(i=ii,**kwargs) for ii in range(len(self.Qsites)))
            else:
                dataall=[]
                for ii in tqdm(range(len(self.Qsites))):
                    datai=self.get_dos(i=ii,**kwargs)
                    dataall.append(datai)
        return np.array(dataall)
        

        
        

        

def mag_hop_square(t,to_site,from_site,phi,lat_spacing):
    coord_i=np.array(from_site.coordinates)/lat_spacing
    coord_f=np.array(to_site.coordinates)/lat_spacing
    dx=(coord_f[0]- coord_i[0])
    ydirection=np.sign(coord_f[1]-coord_i[1])
    nx=coord_i[0]
    return t*np.exp(1j*2*np.pi*phi*nx*dx*ydirection)

def mag_hop_honeycomb(t,to_site,from_site,phi,lat_spacing):
    coord_i=np.array(from_site.coordinates)/lat_spacing
    coord_f=np.array(to_site.coordinates)/lat_spacing
    dy=(coord_f[1]- coord_i[1])
    dy = 0 if np.abs(dy)>1e-6 else 1
    xdirection=np.sign(coord_f[0]-coord_i[0])
    ny=coord_i[1]
    return t*np.exp(1j*2*np.pi*phi*ny*dy*(-1/2)*xdirection)



def onsite_pot(site,Ufunc):
        return Ufunc(site)

def get_dos_i(H,w=None,**kwargs):
    """DOS in one site"""
    from kpmrho import get_dos_i
    if w is None: w = np.linspace(-5.,5.,1000)
    return get_dos_i(H,x=w,**kwargs)
