import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial
from .kpm_solver import kpm_dos, kpm_ldos, kpm_dos_many, kpm_dos_from_ldos
from sklearn.cluster import KMeans

class QuantumSystem:
    def __init__(self, syst, qparams={'Ufunc': lambda x: 0,'phi':0.}):
        """
        Initialize the quantum system.
        
        Parameters:
          Qsites: list of site dictionaries.
          qparams: dictionary of parameters, for example:
              {
                  'Ufunc': <callable that takes a site and returns an onsite energy>,
                  'mag_hop': <callable that takes (site_i, site_j, phi) and returns a hopping amplitude>,
                  'phi': <phase parameter>
              }
        """
        self.Qsites = [syst.sites[idx] for idx in syst.Qsites]
        self.q_to_Q_map={qidx:idx for idx, qidx in enumerate(syst.Qsites)}
        self.all_sites=syst.sites
        self.qparams = qparams.copy()  # Store the parameter dictionary
        self.N = len(self.Qsites)
        self.lattice_type=syst.geometry_params['lattice_type']
        self.lat_spacing=syst.lat_spacing
        self.t=syst.t
        self.H = None
        self.build_hamiltonian()
    
    def build_hamiltonian(self):
        Ufunc = self.qparams['Ufunc']  # This is a function we haven't called yet.
        phi = self.qparams['phi']
        lat_spacing=self.lat_spacing

        hop_func= mag_hop_square if self.lattice_type=="square" else mag_hop_honeycomb
        
        
        self.H = lil_matrix((self.N, self.N), dtype=np.complex128)
        
        for i, site in enumerate(self.Qsites):
            # Diagonal term: call the onsite function with the site.
            self.H[i, i] = onsite_pot(site, Ufunc) 
            for j in site.neighbors:
                if np.abs(self.all_sites[j].coordinates[2]) < 1e-4 and self.all_sites[j].material == 'Qsystem':
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
    def get_energy_bounds(self, margin=0.05):
        """
        Estimate spectral bounds of the Hamiltonian.
        """
        import scipy.sparse.linalg as spla

        H = self.get_hamiltonian()

        try:
            emax = spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
            emin = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        except Exception:
            abs_rowsum = np.abs(H).sum(axis=1).A.ravel()
            bound = abs_rowsum.max()
            emin, emax = -bound, bound

        width = emax - emin
        pad = margin * width

        return emin - pad, emax + pad
    
    def update_qparams(self, new_qparams):
        """
        Update the parameters (such as Ufunc or phi) and rebuild the Hamiltonian.
        """
        self.qparams = {**self.qparams, **new_qparams}
        self.build_hamiltonian()

    def update_U(self,fsc):
        def Ufunc(site):
            return -fsc.Ui[site.id]

        new_qparams = {**self.qparams, "Ufunc": Ufunc}
        self.update_qparams(new_qparams)
        fsc.qparams = dict(new_qparams)
    
    def get_hamiltonian(self):
        return self.H.tocsr()

    def get_dos(self, qparams=None, i=None, w=None, M=512, n_random=10, **kwargs):
        """
        DOS / LDOS from KPM.

        Parameters
        ----------
        qparams : dict or None
            Optional updated Hamiltonian parameters.
        i : None, int, or sequence of ints
            - None: total DOS via stochastic KPM
            - int: LDOS at one site
            - sequence: average LDOS over selected sites
        w : array or None
            Energy grid
        M : int
            Number of Chebyshev moments
        n_random : int
            Number of random vectors for total DOS
        kwargs :
            Forwarded to kpm_solver, e.g. eps=0.05, kernel="jackson", rng=1234
        """
        if qparams is not None:
            self.update_qparams(qparams)

        H = self.get_hamiltonian()

        if i is None:
            return kpm_dos(
                H,
                energies=w,
                num_moments=M,
                num_vectors=n_random,
                **kwargs,
            )

        if np.isscalar(i):
            return kpm_ldos(
                H,
                site_index=int(i),
                energies=w,
                num_moments=M,
                **kwargs,
            )

        site_indices = [int(ii) for ii in i]
        results = kpm_dos_many(
            H,
            site_indices=site_indices,
            energies=w,
            num_moments=M,
            **kwargs,
        )
        avg = np.mean([rho for _, rho in results], axis=0)
        return np.asarray(w, dtype=float), avg

    def get_ldos(self, fsc, qparams=None, approx="TF", Ncore=0, M=512, n_random=8, **kwargs):
        emin, emax = self.get_energy_bounds()
        Erange = np.linspace(emin, emax, int(len(self.Qsites) / 2))

        if qparams is not None:
            self.update_qparams(qparams)

        if approx == "TF":
            bulk_dos = self.get_dos(w=Erange, M=M, n_random=n_random)
            dataall = [bulk_dos for _ in range(len(self.Qsites))]

        elif approx == "kmeanssample":
            dataall = self.sample_ldos(fsc, Ncore=Ncore, M=M, **kwargs)

        else:
            H = self.get_hamiltonian()

            if Ncore > 1:
                dataall = Parallel(n_jobs=Ncore)(
                    delayed(kpm_ldos)(
                        H,
                        site_index=ii,
                        energies=Erange - fsc.Ui[fsc.Qsites][ii],
                        num_moments=M
                    )
                    for ii in range(len(self.Qsites))
                )
            else:
                dataall = []
                for ii in tqdm(range(len(self.Qsites))):
                    datai = kpm_ldos(
                        H,
                        site_index=ii,
                        energies=Erange - fsc.Ui[fsc.Qsites][ii],
                        num_moments=M
                    )
                    dataall.append(datai)

        return np.array(dataall, dtype=object)

    def sample_ldos(self, fsc, num_sample=20, Ncore=1, M=512, u_weight=2.0,return_groups=False, **kwargs):
        from sklearn.cluster import KMeans
        import numpy as np

        emin, emax = self.get_energy_bounds()
        Erange = np.linspace(emin, emax, int(len(self.Qsites) / 2))

        qprime_ids = list(fsc.Qprime)
        coords = np.array([fsc.sites[idx].coordinates[:2] for idx in qprime_ids])
        Uvals = np.array([fsc.Ui[idx] for idx in qprime_ids])

        num_sample = min(num_sample, len(coords))

        # standardize features
        x = coords[:, 0]
        y = coords[:, 1]
        u = Uvals.copy()

        def safe_standardize(arr):
            s = arr.std()
            if s < 1e-14:
                return np.zeros_like(arr)
            return (arr - arr.mean()) / s

        xz = safe_standardize(x)
        yz = safe_standardize(y)
        uz = safe_standardize(u) * u_weight

        features = np.column_stack([xz, yz, uz])

        kmeans = KMeans(n_clusters=num_sample, n_init=10)
        labels = kmeans.fit_predict(features)

        site_in_b = []
        for k in range(num_sample):
            idx = np.where(labels == k)[0]
            if len(idx) > 0:
                site_in_b.append(idx)

        Uis = fsc.Ui[fsc.Qsites]
        Qp_in_Q_map = fsc.Qp_in_Q.copy()
        H = self.get_hamiltonian()

        def calculate_ldos(indices):
            # choose representative site closest to cluster center in feature space
            cluster_features = features[indices]
            center = cluster_features.mean(axis=0)
            local_idx = np.argmin(np.linalg.norm(cluster_features - center, axis=1))
            rep_site = indices[local_idx]

            qidx = Qp_in_Q_map[rep_site]
            energies = Erange - Uis[qidx]

            E, rho = kpm_ldos(
                H,
                site_index=qidx,
                energies=energies,
                num_moments=M,
                **kwargs
            )

            E = E + Uis[qidx]
            return np.array([E, rho])

        if Ncore > 1:
            bin_ldos = Parallel(n_jobs=Ncore)(
                delayed(calculate_ldos)(indices) for indices in site_in_b
            )
        else:
            bin_ldos = [calculate_ldos(indices) for indices in site_in_b]

        dataall = []
        for bidx, indices in enumerate(site_in_b):
            dataall.append(np.array([bin_ldos[bidx] for _ in range(len(indices))]))

        sortidx = np.argsort(np.concatenate(site_in_b))
        ldos_out = np.concatenate(dataall)[sortidx]

        if return_groups:
            rep_sites = []
            for indices in site_in_b:
                cluster_features = features[indices]
                center = cluster_features.mean(axis=0)
                local_idx = np.argmin(np.linalg.norm(cluster_features - center, axis=1))
                rep_sites.append(indices[local_idx])

            group_info = {
                "labels": labels,
                "site_in_b": site_in_b,
                "rep_sites": np.array(rep_sites, dtype=int),
                "coords": coords,
            }
            return ldos_out, group_info

        return ldos_out
    


        


        

        
        

        

def mag_hop_square(t,to_site,from_site,phi,lat_spacing):
    coord_i = np.array(from_site.coordinates) / lat_spacing
    coord_f = np.array(to_site.coordinates) / lat_spacing

    dx = coord_f[0] - coord_i[0]
    dy = coord_f[1] - coord_i[1]
    x = coord_i[0]

    return t * np.exp(1j * 2 * np.pi * phi * x * dy)
    #coord_i=np.array(from_site.coordinates)/lat_spacing
    #coord_f=np.array(to_site.coordinates)/lat_spacing
    #dx=(coord_f[0]- coord_i[0])
    #ydirection=np.sign(coord_f[1]-coord_i[1])
    #nx=coord_i[0]
    #return t*np.exp(1j*2*np.pi*phi*nx*ydirection)

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

