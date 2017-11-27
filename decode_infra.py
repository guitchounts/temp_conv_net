
# coding: utf-8

# In[2]:

import os
import h5py
import numpy as np
import pandas as pd
from data_helpers import grouper
from TempConv import determine_fit


# In[9]:

nn_params = {
    'bs' : 256,
    'eps' : 15,
    'lr' : 0.0005,
    'kernel' : 2,
    'nb_filter' : 5,
    'window' : 60,
    'offset' : 10,
    'nb_test' : 3,
    'nb_trains' : 5,
    'verbose' : False
}


# In[4]:

## get and format data
lfp_file = h5py.File('datasets/GRat31_636061_lfp_power.hdf5', 'r')
neural_data = np.asarray(lfp_file['lfp_power']) # iterate through powerbands
tetrodes = grouper(neural_data, 24)

head_signals_h5 = h5py.File('datasets/GRat31_636061_all_head_data.hdf5', 'r')
idx_start, idx_stop = [6,9]
head_signals = np.asarray([np.asarray(head_signals_h5[key]) for key in head_signals_h5.keys()][0:9]).T[:,idx_start:idx_stop]
print('head_signals shape: ', head_signals.shape)

head_signals_keys = list(head_signals_h5.keys())[0:9][idx_start:idx_stop]
head_signals_int = ['yaw_abs', 'roll_abs', 'pitch_abs']
print('head_signals_keys intuitive: ', head_signals_int)


# In[10]:

stats = {}


# In[12]:

# iterate Xs
for tetrode_idx in range(tetrodes.shape[0]):
    tetrode = tetrodes[tetrode_idx].T
    
    #if tetrode_idx >= 1: break
    
    # iterate ys
    for head_signal_idx in range(head_signals.shape[1]):
        R2r_arr = {
            'R2s' : [],
            'rs' : []
        }
        
        for i in range(nn_params['nb_trains']): # replace with k-fold? n k-folds?
            head_signal = head_signals[:,head_signal_idx]
            R2, r = determine_fit(tetrode, head_signal, [head_signals_int[head_signal_idx]], nn_params)
            
            R2r_arr['R2s'].append(R2[0])
            R2r_arr['rs'].append(r[0])
        
        stats['tetrode_{}_head_signal_{}'.format(tetrode_idx, head_signal_idx)] = R2r_arr


# In[ ]:

print(stats)
