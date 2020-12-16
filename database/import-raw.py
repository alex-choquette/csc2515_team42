# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 02:34:32 2020

@author: Chris

extract raw data from QM9 files
"""

from schnetpack.datasets import QM9
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import json


#%% import QM9 database
qm9data = QM9('./qm9.db', download=True)


#%% convert to different representations
rep = pd.DataFrame(data = None, columns=qm9data[0].keys())
row = {}
i = 0

for n in tqdm(range(len(qm9data))):
    print(i)
    datum = qm9data[n]
    
    # convert tensors to numpy arrays
    for k, v in datum.items():
        row[k] = v.numpy()

    # append row to dataframe
    rep = rep.append(row, ignore_index=True)
    
    i+=1

rep.to_json('qm9_raw.json')

