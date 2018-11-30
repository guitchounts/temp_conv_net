from scipy import stats, signal
import numpy as np
from itertools import zip_longest
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn import preprocessing
sns.set_style('white')

def sample_dx_uniformly(derivative,num_points=50000):
    ################### sample the dx distribution evenly: ####################
    #print('len(derivative), num_points = ', len(derivative), num_points)
    if len(derivative) < num_points:
        num_points = len(derivative)

    derivative = np.squeeze(derivative)
    bins = 10000
    hist,edges = np.histogram(derivative,bins=bins,normed=True)
    
    bins_where_values_from = np.searchsorted(edges,derivative)
    
    bin_weights = 1/(hist/sum(hist))
    
    inv_weights = bin_weights[bins_where_values_from-1]
    
    dx_idx = np.arange(0,len(derivative),1)

    sampled_dx_idx = np.random.choice(dx_idx,size=num_points,replace=False,p =inv_weights/sum(inv_weights)  )

    sampled_dx = np.random.choice(derivative,size=num_points,replace=False,p =inv_weights/sum(inv_weights)  )


    f,axarr = plt.subplots(2,dpi=600,sharex=True)

    axarr[0].hist(derivative,bins=200)
    axarr[0].set_ylabel('d_yaw \n original')

    axarr[1].hist(sampled_dx,bins=200)
    axarr[1].set_ylabel('d_yaw \n resampled')

    sns.despine(left=True,bottom=True)

    f.savefig('resampled_original_histograms.pdf')

    return sampled_dx_idx

def pass_filter(ephys,freq_range,filt_order = 4,filt_type='bandpass',fs=10.):
    # design Elliptic filter:

    [b,a] = signal.butter(filt_order,[freq/(fs/2) for freq in freq_range],btype=filt_type)
    
    filtered_trace = signal.filtfilt(b,a,ephys,axis=0)
    return filtered_trace

def split_data(X, y, test_size, standardize=True,shuffle=False):
    test_size_idx = int(test_size * X.shape[0])
    X_train, X_test, y_train, y_test = X[:-test_size_idx], X[-test_size_idx:], y[:-test_size_idx], y[-test_size_idx:]
    # X shape = e.g. (719800, 200, 16)
    if standardize:
        #Z-score "X" inputs. 
        X_train_mean = np.nanmean(X_train, axis=0)
        X_train_std = np.nanstd(X_train, axis=0)
        if 0 in X_train_std:
            print('Zero values encountered in X_train_std. Zero-centering but not Z-scoring.')
            X_train = X_train - X_train_mean
            X_test = X_test - X_train_mean
        else:
            X_train = (X_train - X_train_mean) / X_train_std
            X_test = (X_test - X_train_mean) / X_train_std

        #Zero-center outputs
        #y_train_mean = np.mean(y_train, axis=0)
        #y_train = y_train - y_train_mean
        #y_test = y_test - y_train_mean
    
    if shuffle:
        print('!!!!!!!!!!!!!!!!!! SHUFFLING X_test !!!!!!!!!!!!!!!!!!')
        #X_test = X_test.reshape(-1,shuffle_chunk_size)
        np.random.shuffle(X_test)
        #X_test.flatten()


    return X_train, X_test, y_train, y_test

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return np.asarray(list(zip_longest(*args, fillvalue=fillvalue)))

def make_timeseries_instances(X, y, window_size, offset):
    X = np.asarray(X)
    y = np.asarray(y)
    assert 0 < window_size <= X.shape[0]
    assert X.shape[0] == y.shape[0]
    X = np.atleast_3d(np.array([X[start:start + window_size] for start in range(0, X.shape[0] - window_size)]))
    y = y[window_size-offset:-offset]
    return X, y

def timeseries_shuffler(X, y, series_length, padding):
    """shuffle time series data, chunk by chunk into even and odd bins, discarding a pad
    between each bin.

    Keyword arguments:
    X -- input dataset
    y -- output dataset
    series_length -- length of chunks to bin by
    padding -- pad to discard between each chunk.
    """
    X_even = []
    X_odd = []
    y_even = []
    y_odd = []

    # state variable control which bin to place data into
    odd = False
    
    for i in range(X.shape[0]):
        # after series_length + padding, switch odd to !odd
        if (i%(series_length+padding)) == 0:
            odd = not odd

        # only add to bin during the series period, not the padding period
        if (i%(series_length+padding))<series_length:
            
            # put X[i] and y[i] into even/odd bins
            if odd:
                X_odd.append(X[i])
                y_odd.append(y[i])
            else:
                X_even.append(X[i])
                y_even.append(y[i])

    # concatenate back together
    X_even.extend(X_odd)
    y_even.extend(y_odd)
    
    # put them back into np.arrays
    X_shuffled = np.asarray(X_even)
    y_shuffled = np.asarray(y_even)
    
    return X_shuffled, y_shuffled

###$$ GET_SPIKES_WITH_HISTORY #####
def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    print('******************************** Getting Spikes with History *************************************')

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    print('num_examples, num_neurons, surrounding_bins = ', num_examples, num_neurons, surrounding_bins)
    
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    #X[:] = np.NaN

    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;
    return X
