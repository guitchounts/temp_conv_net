
# coding: utf-8

# In[2]:

import os,sys
import h5py
import numpy as np
import pandas as pd
from data_helpers import grouper
from TempConv import determine_fit
from scipy import stats,signal
from skimage import exposure
# In[9]:



def filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):
    
    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)
    
    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace

def run_decoding(lfp_path,head_path,nn_params):

    ## get and format data
    lfp_file = h5py.File(lfp_path, 'r')
    print('lfp_file keys:',lfp_file.keys())
    data_name = list(lfp_file.keys())[0]





    neural_data = np.asarray(lfp_file[data_name]) # iterate through powerbands



    print('Shape of neural data, as loaded: ', neural_data.shape)
    if neural_data.shape[0] > neural_data.shape[1]:   #### ephys should be channels x samples
        neural_data = neural_data.T


    if data_name.find('lfp_power') == 1:
        ### take average of LFP bands in each tetrode: 
        avgd_neural_data = np.empty([64,neural_data.shape[1]])
        fs = 10.
    else: 
        fs = 100.

    ##### shuffle control:  neural_data = np.random.permutation(neural_data.T).T

    tetrodes = grouper(neural_data, neural_data.shape[0])


    ### bad electrode control?
    ### tetrodes = tetrodes[:,144:162,:]

    print(tetrodes.shape)

    head_signals_h5 = h5py.File(head_path, 'r')
    idx_start, idx_stop = [0,9]
    head_signals = np.asarray([np.asarray(head_signals_h5[key]) for key in head_signals_h5.keys()][0:9]).T[:,idx_start:idx_stop]
    print('head_signals shape: ', head_signals.shape)


    
    xyz = filter(np.sqrt(head_signals[:,0]**2 + head_signals[:,1]**2 + head_signals[:,2]**2     ),[1],filt_type='lowpass',fs=fs)

    # dx_neg = np.empty(head_signals[:,3].shape)
    # dx_pos = np.empty(head_signals[:,3].shape)
    # dx = head_signals[:,3]
    # dx_neg[np.where(dx < 0)[0]] = dx[np.where(dx < 0)[0]]

    # dx_pos[np.where(dx > 0)[0]] = dx[np.where(dx > 0)[0]]


    # dx = np.gradient(filter(  np.rad2deg(np.unwrap(np.deg2rad(head_signals[:,6]))),[1],filt_type='lowpass',fs=fs  )      )
    # dy = np.gradient(filter(  np.rad2deg(np.unwrap(np.deg2rad(head_signals[:,7]))),[1],filt_type='lowpass',fs=fs  )      )
    # dz = np.gradient(filter(  np.rad2deg(np.unwrap(np.deg2rad(head_signals[:,8]))),[1],filt_type='lowpass',fs=fs  )      )
    
    # filt_roll = filter(head_signals[:,7],[1.],fs=fs,filt_type='lowpass')
    # filt_pitch = filter(head_signals[:,8],[1.],fs=fs,filt_type='lowpass')

    # lowpass_dy = np.gradient(filt_roll)
    # lowpass_dz = np.gradient(filt_pitch)

    head_signals = np.vstack([head_signals[:,6],head_signals[:,7],head_signals[:,8], xyz]).T
    #head_signals = np.vstack([dx,dy,dz]).T
    #head_signals_int = ['left','right']  



    head_signals_keys = list(head_signals_h5.keys())[0:9][idx_start:idx_stop]
    head_signals_int = ['yaw_abs', 'roll_abs', 'pitch_abs', 'total_acc']
    #head_signals_int = ['d_yaaw', 'd_roll','d_pitch']

    print('head_signals_keys intuitive: ', head_signals_int)

    ## limit signals to 1e6 samples:
    limit = int(1e6)
    if neural_data.shape[1] > limit:
        print('Reducing Data Size Down to %d Samples' % limit)
        tetrodes  = tetrodes[:,:,0:limit]
        head_signals = head_signals[0:limit,:]
        
    print('The SHAPE of tetrodes and head_signals = ', tetrodes.shape,head_signals.shape)
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
                R2, r = determine_fit(tetrode, head_signal, [head_signals_int[head_signal_idx]], nn_params,)
                
                R2r_arr['R2s'].append(R2[0])
                R2r_arr['rs'].append(r[0])
            
            stats['tetrode_{}_head_signal_{}'.format(tetrode_idx, head_signal_idx)] = R2r_arr


    # In[ ]:

    print(stats)



if __name__ == "__main__":

    nn_params = {
        'bs' : 256,
        'eps' : 35,
        'lr' : 0.0005,
        'kernel' : 2,
        'nb_filter' : 5,
        'window' : 60,
        'offset' : 30,
        'nb_test' : 1,
        'nb_trains' : 1,
        'verbose' : False,
        'id' : 3
    }

    lfp_path = sys.argv[1]
    head_path = sys.argv[2]

    run_decoding(lfp_path,head_path,nn_params)
