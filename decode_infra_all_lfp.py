
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
from functools import reduce
# In[9]:


def zero_runs(a):  # from link
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0] #.reshape(-1, 2)
    if len(ranges) > 0:
        
        return ranges[0]
    else:
        return a.shape[0]

def filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):

    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)

    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace

def get_head_stop(head_data): ## head_data.shape = e.g. (1000000, 4)
    all_diffs = []
    head_names = range(head_data.shape[1])  #['ox','oy','oz','ax','ay','az']
    for head_name in head_names:
        diffs =np.where(np.diff(head_data[:,head_name]) == 0 )[0] ##  zero_runs(np.diff(head_data[:,head_name])) ###
        all_diffs.append(diffs)
        print('Getting start/stop coordinates for %s. Shape of diffs = ' % (head_name), diffs.shape)

    all_zeros = reduce(np.intersect1d, (all_diffs))
    #stop = np.min(all_diffs)
    if len(all_zeros) == 0:
        stop = head_data.shape[0] + 1
    else:
        stop = all_zeros[0]

    
    return stop

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

    tetrodes = grouper(neural_data, neural_data.shape[0])


    ### bad electrode control?
    ### tetrodes = tetrodes[:,144:162,:]

    print(tetrodes.shape)

    head_signals_h5 = h5py.File(head_path, 'r')
    idx_start, idx_stop = [0,9]
    head_signals = np.asarray([np.asarray(head_signals_h5[key]) for key in head_signals_h5.keys()][0:9]).T[:,idx_start:idx_stop]
    print('head_signals shape: ', head_signals.shape)



    xyz = filter(np.sqrt(head_signals[:,0]**2 + head_signals[:,1]**2 + head_signals[:,2]**2     ),[1],filt_type='lowpass',fs=fs)
    ## check for NaNs in xyz and replace them with zeros:
    xyz[np.where(np.isnan(xyz))[0]] = 0.


    
    
    head_signals = np.hstack([head_signals,np.atleast_2d(xyz).T])
    # dx_neg = np.empty(head_signals[:,3].shape)
    # dx_pos = np.empty(head_signals[:,3].shape)
    # dx = head_signals[:,3]
    # dx_neg[np.where(dx < 0)[0]] = dx[np.where(dx < 0)[0]]

    # dx_pos[np.where(dx > 0)[0]] = dx[np.where(dx > 0)[0]]

    if 'dx' in nn_params.keys():
        if nn_params['dx'] == 1:
            dx = np.gradient(filter(  np.rad2deg(np.unwrap(np.deg2rad(head_signals[:,6]))),[1],filt_type='lowpass',fs=fs  )      )
            dy = np.gradient(filter(  np.rad2deg(np.unwrap(np.deg2rad(head_signals[:,7]))),[1],filt_type='lowpass',fs=fs  )      )
            dz = np.gradient(filter(  np.rad2deg(np.unwrap(np.deg2rad(head_signals[:,8]))),[1],filt_type='lowpass',fs=fs  )      )
            head_signals = np.vstack([dx,dy,dz]).T
            head_signals_int = ['dyaw', 'droll', 'dpitch']
            limit = int(1e7)
    else:

        
        ## see in params if you'd like to only decode a specific signal (e.g. yaw or roll or pitch)
        if 'decode_signals' in nn_params.keys():

            head_signals_int = nn_params['decode_signals'].split(',')
            idx = []
            if 'yaw_abs' in head_signals_int:
                idx.append(6)                
            if 'roll_abs' in head_signals_int:
                idx.append(7)                
            if 'pitch_abs' in head_signals_int:
                idx.append(8)                
            if 'total_acc' in head_signals_int:
                idx.append(9)
            if 'yaw_tree' in head_signals_int:
                idx.append(6)  
            if 'yaw_mse' in head_signals_int:
                idx.append(6)      
    
            
            head_signals = np.vstack([head_signals[:,x] for x in idx ]).T
            print('after decode_signals, head_signals.shape = ',head_signals.shape)

        else:

            head_signals = np.vstack([head_signals[:,6],head_signals[:,7],head_signals[:,8], head_signals[:,9]]).T
            print('after NOT decode_signals, head_signals.shape = ',head_signals.shape)
            head_signals_int = ['yaw_abs', 'roll_abs', 'pitch_abs', 'total_acc']
        

        limit = int(1e6)
    # filt_roll = filter(head_signals[:,7],[1.],fs=fs,filt_type='lowpass')
    # filt_pitch = filter(head_signals[:,8],[1.],fs=fs,filt_type='lowpass')

    # lowpass_dy = np.gradient(filt_roll)
    # lowpass_dz = np.gradient(filt_pitch)

    
    
    #head_signals_int = ['left','right']



    head_signals_keys = list(head_signals_h5.keys())[0:9][idx_start:idx_stop]
    
    #head_signals_int = ['d_yaaw', 'd_roll','d_pitch']

    #print('head_signals_keys intuitive: ', head_signals_int)

    ## limit signals to 1e6 samples:
    
    # if neural_data.shape[1] > limit:
    #     print('Reducing Data Size Down to %d Samples' % limit)
    #     tetrodes  = tetrodes[:,:,0:limit]
    #     head_signals = head_signals[0:limit,:]

    print('The SHAPE of tetrodes and head_signals = ', tetrodes.shape,head_signals.shape)
    # In[10]:
    two_hour_lim = int(100*60*60*2)

    ## in case the BNO recording failed and recorded a bunch of zeros, cut out those zeros from the end:
    start,stop = 0,get_head_stop(head_signals)
    head_signals = head_signals[start:stop,:]
    tetrodes = tetrodes[:,:,start:stop]
    print('head_signals shape after start,stop = ', head_signals.shape)
    num_chunks = max(1,int(head_signals.shape[0] / two_hour_lim)) ## how many two-hour chunks of decoding can we do using this dataset?


    ## lowpass filter:
    for x in range(head_signals.shape[1]):
        print('Filtering head signal %s' % head_signals_int[x])
        head_signals[:,x] = filter(head_signals[:,x],[1],filt_type='lowpass',fs=fs)



    # split tetrodes and head data into chunks:
    chunk_indexes = [two_hour_lim*i for i in range(num_chunks+1)] ## get indexes like [0, 720000] [720000, 1440000] [1440000, 2160000]

    chunk_indexes = [[v, w] for v, w in zip(chunk_indexes[:-1], chunk_indexes[1:])] # reformat to one list
    print('chunk_indexes = ', chunk_indexes)
    all_tetrodes = [tetrodes[:,:,chunk_indexes[chunk][0]:chunk_indexes[chunk][1]] for chunk in range(num_chunks)  ] ## list of 1x16x720000 chunks
    all_head_signals = [head_signals[chunk_indexes[chunk][0]:chunk_indexes[chunk][1],:] for chunk in range(num_chunks)  ]
    print('all_head_signals[chunk] shape after chunking.   =', all_head_signals[0].shape)


    stats = {}

    model_type = config['config']['model_type']

    # In[12]:

     # iterate Xs

    for chunk in range(num_chunks):

        stats = {}
        for tetrode_idx in range(tetrodes.shape[0]): ### should be range(2) for tetrodes split into left and right hemispheres. first = RH, second = LH.
            tetrode = all_tetrodes[chunk][tetrode_idx].T  # tetrodes[tetrode_idx].T

            chunk_save_dir = save_dir + str(chunk) + '/' ### make a /1/left/ and /1/right, /2/left/ and /2/right etc subdir for saving
            if not os.path.exists(chunk_save_dir):
                os.makedirs(chunk_save_dir)

           

            # iterate ys
            for head_signal_idx in range(head_signals.shape[1]): ## four for yaw, roll, pitch, and total_acc
                R2r_arr = {
                    'R2s' : [],
                    'rs' : []
                }

                y_key = [head_signals_int[head_signal_idx]]
                num_trains = range(nn_params['nb_trains'])
                
                print('all_head_signals[chunk] shape   =', all_head_signals[chunk].shape)

                head_signal = all_head_signals[chunk][:,head_signal_idx] ###  head_signals[:,head_signal_idx]
                
                print('head_signal shape before compleixfiying  =', head_signal.shape)

                if  any("yaw_complex" in s for s in y_key) or any("yaw_tree" in s for s in y_key): # was: for s in head_signals_int
                    print(y_key)
                    print('Modeling YAW as complex number!!')
                    head_signal = np.exp( 1j * np.deg2rad(head_signal) )
                    print('complex head_signal shape =', head_signal.shape)
                    head_signal = [head_signal.real, head_signal.imag]

                    custom_loss = 0
                    #y_key = ['yaw_real','yaw_imag']
                    #num_trains = range(2)
                else:
                    print('Modeling %s Not as complex number' % y_key)
                    head_signal = head_signal

                    if any("yaw_mse" in s for s in y_key):
                        print('Evaluating %s Model. Custom Loss = 1 ' % y_key)
                        custom_loss = 1
                    else:
                        custom_loss=0

                #for i in num_trains:
                print('***************** Running Decoding on Chunk %d' % (chunk))

                #print('head_signal.shape = ', head_signal.shape)
                
                if  any("yaw_complex" in s for s in y_key):
                    new_keys = ['yaw_real','yaw_imag']
                    for j in range(2):
                        R2, r = determine_fit(tetrode, head_signal[j], [new_keys[j]], nn_params, chunk_save_dir,model_type=model_type,custom_loss=custom_loss)
                        R2r_arr['R2s'].append(R2[0])
                        R2r_arr['rs'].append(r[0])
                
                elif  any("yaw_tree" in s for s in y_key):
                    #new_keys = ['yaw_real','yaw_imag']
                    #model_type = 'tree'
                    R2, r = determine_fit(tetrode, np.vstack(head_signal).T, y_key, nn_params, chunk_save_dir,model_type='tree',custom_loss=custom_loss)
                    R2r_arr['R2s'].append(R2[0])
                    R2r_arr['rs'].append(r[0])


                else:
                    R2, r = determine_fit(tetrode, head_signal, y_key, nn_params, chunk_save_dir,model_type=model_type,custom_loss=custom_loss)

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
