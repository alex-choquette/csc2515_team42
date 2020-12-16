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
rep_init ={'SOAP': descriptors.SOAP(species=species, periodic=False, 
                       rcut=rcut, nmax=5, lmax=3)
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
           
rep = pd.DataFrame(data=None, columns=list(rep_init.keys())+props)

for n in tqdm(range(size)):
    datum = {}
    for k, v in rep_init.items():    
        datum[k] = v.create(qm9data.get_atoms(idx=n))
    
    # diagonalize cm
    #datum['CoulombMatrix'] = np.linalg.eigh(datum['CoulombMatrix'])
    
    for col in props:
        datum[col] = qm9data[n][col].numpy()
        
    rep = rep.append(datum, ignore_index=True)

rep.to_json('reps_SOAP.json')