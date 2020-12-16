# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 04:26:29 2020

@author: Chris

Convert QM9 from xyz to useful feature database
"""

from schnetpack.datasets import QM9
from dscribe import descriptors
import pandas as pd
import torch
import h5py
import numpy as np
from tqdm import tqdm
import json


#%% import QM9 database
props = ['energy_U0', 'energy_U', 'homo', 'lumo', 'gap', 'enthalpy_H', 
         'free_energy', 'heat_capacity']
qm9data = QM9('./qm9.db', download=True, load_only=props)
size = len(qm9data)


#%% convert to different representations
species = ['H', 'C', 'N', 'O', 'F']
rcut = 6.0
rep_init ={'CoulombMatrix': descriptors.CoulombMatrix(n_atoms_max=40, flatten=False)#,
           #'SOAP': descriptors.SOAP(species=species, periodic=False, 
           #            rcut=rcut, nmax=8, lmax=6)
          }
'''
           'ACSF': descriptors.SOAP(species=species, rcut=rcut,
                        g2_params = [[]],
                        g3_params = [],
                        g4_params = [[[]]],
                        g5_params = [[[]]]
                        )
           'MBTR': descriptors.MBTR(species=species, periodic=False,
                        flatten=True,
                        k1 = {"geometry": {"function": "atomic_number"},
                              "grid": {"min": 1, "max": 10, "sigma": 0.1, "n": 50}
                             },
                        k2 = {"geometry": {"function": "inverse_distance"},
                              "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 50},
                              "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                             },
                        k3 = {"geometry": {"function": "angle"},
                              "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
                              "weighting" : {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                             }
                        )         
'''

# add numeric data to HDF5
f = h5py.File('qm9-rep.hdf5', 'w')
for col in props:
    data = []
    
    for n in tqdm(range(size), desc=col):
        data.append(qm9data[n][col].numpy())
        
    f.create_dataset(col, data=data)


for k, v in rep_init.items():    
    data = []
    
    for n in tqdm(range(size), desc=k):
        data.append(v.create(qm9data.get_atoms(idx=n)))
    
    f.create_dataset(k, data=data)
    
    # diagonalize cm
    if k == 'CoulombMatrix':
        f.create_dataset('CoulombEigh', data=np.linalg.eigh(data)[0])
    
f.close()
    