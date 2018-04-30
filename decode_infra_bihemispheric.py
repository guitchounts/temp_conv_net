
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
import json
# In[9]:



def filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):
    
    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)
    
    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace

def run_decoding(lfp_path,head_path,nn_params,save_dir):

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

    tetrodes = grouper(neural_data, int(neural_data.shape[0] / 2)) ### GROUPING INTO LEFT and RIGHT hemispheres - first 8 and last 8 tetrodes
    ### 0:8 = RH, 8:16 = LF


    ### bad electrode control?
    ### tetrodes = tetrodes[:,144:162,:]

    print(tetrodes.shape)

    head_signals_h5 = h5py.File(head_path, 'r')
    idx_start, idx_stop = [0,9]
    head_signals = np.asarray([np.asarray(head_signals_h5[key]) for key in head_signals_h5.keys()][0:9]).T[:,idx_start:idx_stop]
    print('head_signals shape: ', head_signals.shape)


    
    xyz = filter(np.sqrt(head_signals[:,0]**2 + head_signals[:,1]**2 + head_signals[:,2]**2     ),[1],filt_type='lowpass',fs=fs)

    ## lowpass filter:
    for x in range(6,9):
        print('Filtering head signal %s' % list(head_signals_h5.keys())[x])
        head_signals[:,x] = filter(head_signals[:,x],[1],filt_type='lowpass',fs=fs)


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
    # limit = int(1e6)
    # if neural_data.shape[1] > limit:
    #     print('Reducing Data Size Down to %d Samples' % limit)
    #     tetrodes  = tetrodes[:,:,0:limit]
    #     head_signals = head_signals[0:limit,:]
        
    print('The SHAPE of tetrodes and head_signals = ', tetrodes.shape,head_signals.shape) 
    # e.g.  (1, 16, 1000000) (1000000, 4)
    ## two-hour chunks at 100 samples / sec = 
    two_hour_lim = int(100*60*60*2)
    num_chunks = int(2e6 / two_hour_lim) ## how many two-hour chunks of decoding can we do using this dataset?

    # split tetrodes and head data into chunks:
    chunk_indexes = [two_hour_lim*i for i in range(num_chunks+1)] ## get indexes like [0, 720000] [720000, 1440000] [1440000, 2160000]

    chunk_indexes = [[v, w] for v, w in zip(chunk_indexes[:-1], chunk_indexes[1:])] # reformat to one list

    hemispheres = ['right', 'left']

    all_tetrodes = [tetrodes[:,:,chunk_indexes[chunk][0]:chunk_indexes[chunk][1]] for chunk in range(num_chunks)  ] ## list of 1x16x720000 chunks 
    all_head_signals = [head_signals[chunk_indexes[chunk][0]:chunk_indexes[chunk][1],:] for chunk in range(num_chunks)  ]




    model_type = config['config']['model_type']

    

    # iterate Xs

    for chunk in range(num_chunks):
            
        stats = {}            
        for tetrode_idx in range(tetrodes.shape[0]): ### should be range(2) for tetrodes split into left and right hemispheres. first = RH, second = LH. 
            tetrode = all_tetrodes[chunk][tetrode_idx].T  # tetrodes[tetrode_idx].T
            
            left_right_save_dir = save_dir + chunk + '/' + hemispheres[tetrode_idx] + '/' ### make a /1/left/ and /1/right, /2/left/ and /2/right etc subdir for saving
            if not os.path.exists(left_right_save_dir):
                os.makedirs(left_right_save_dir)

            # iterate ys
            for head_signal_idx in range(head_signals.shape[1]): ## four for yaw, roll, pitch, and total_acc
                R2r_arr = {
                    'R2s' : [],
                    'rs' : []
                }
                
                for i in range(nn_params['nb_trains']):
                    head_signal = all_head_signals[chunk][:,head_signal_idx] ###  head_signals[:,head_signal_idx]

                    print('***************** Running Decoding on Chunk %d, %s Hemisphere' % (chunk,hemispheres[tetrode_idx]))
                    

                    R2, r = determine_fit(tetrode, head_signal, [head_signals_int[head_signal_idx]], nn_params, left_right_save_dir,model_type=model_type)
                    
                    R2r_arr['R2s'].append(R2[0])
                    R2r_arr['rs'].append(r[0])
                
                stats['tetrode_{}_head_signal_{}'.format(tetrode_idx, head_signal_idx)] = R2r_arr
                print(stats)

    



if __name__ == "__main__":

    # nn_params = {
    #     'bs' : 256,
    #     'eps' : 35,
    #     'lr' : 0.0005,
    #     'kernel' : 2,
    #     'nb_filter' : 5,
    #     'window' : 100,
    #     'offset' : 50,
    #     'nb_test' : 1,
    #     'nb_trains' : 1,
    #     'verbose' : False,
    #     'id' : 3
    # }

    # lfp_path = sys.argv[1]
    # head_path = sys.argv[2]
    config_file = sys.argv[1]

    with open(config_file) as json_data_file:
        config = json.load(json_data_file)


# cd  "C:\Users\Grigori Guitchounts\Dropbox (coxlab)\Ephys\Data" \ &&
# cd .\636505099725591062\ &&
# cd &&
# mkdir 031218_mua && cd .\031218_mua &&
# python "C:\Users\Grigori Guitchounts\Documents\GitHub\temp_conv_net\decode_infra_all_lfp.py" ..\mua_firing_rates_100hz.hdf5 ..\all_head_data_100hz.hdf5 &&

    #### assuming we're in the GratXXX directory. 
    input_file_path = os.getcwd()
    all_files = []
    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                all_files.append(file)
    all_files = np.asarray(all_files)

    for fil in all_files:

        save_dir = './' + fil + '/' + config['config']['experiment'] + '/'

        

        neural_path = './' + fil + '/' + config['config']['neural_data']
        head_path = './' + fil + '/' + config['config']['head_data']
        print('************************************************************************************')
        print('*************************** Running Decoding on %s *********************************' % save_dir)
        print('*************************** Neural Data from %s ************************************' % neural_path)
        print('*************************** Head Data from %s **************************************' % head_path)
        print('************************************************************************************')
        

        if os.path.exists(head_path): #### make sure the experiment directory has the neural/head data:
            if os.path.exists(neural_path):
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                run_decoding(neural_path,head_path,config['nn_params'],save_dir)



















