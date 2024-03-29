import numpy as np
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from data_helpers import pass_filter, split_data, make_timeseries_instances, timeseries_shuffler,get_spikes_with_history
from metrics_helper import do_the_thing
import keras.backend as K
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

def modified_mse(y_true, y_pred): #### modified MSE loss function for absolute yaw data (0-360 values wrap around)
    

    #y_true = y_true * y_std + y_mean ### y_train_mean,y_train_std are GLOBALS ??? 
    #y_pred = y_pred * y_std + y_mean

    mod_square =  K.square(K.abs(y_pred - y_true) - 360) ### hack 2.1 = (360 - np.mean(ox)) / np.std(ox) 2.1086953197291871
    raw_square =  K.square(y_pred - y_true)
    better = K.minimum(mod_square,raw_square)
    return K.mean(better,axis= -1)

def get_turn_idx(dx):
    dx = np.squeeze(dx)
    print('dx.shape = ', dx.shape)
    print('dx max = ', np.max(dx))

    dx_neg = np.zeros(dx.shape)
    dx_pos = np.zeros(dx.shape)

    #dx_neg[dx < 0] = dx[dx < 0]
    #dx_pos[dx > 0] = dx[dx > 0]
    print(dx_neg[np.where(dx<0)[0]])
    dx_neg[np.where(dx<0)[0]] = dx[np.where(dx<0)[0]]
    dx_pos[np.where(dx>0)[0]] = dx[np.where(dx>0)[0]]

    left = dx_pos**2
    right = dx_neg**2
    print('here are some left, right turns: ', left[0:5],right[0:5])
    turns = np.zeros(dx.shape)
    print('turns shape = ', turns.shape)
    thresh = 0.25

    turns[left > thresh] = 1

    turns[right > thresh] = -1

    x =np.where(np.diff(turns) )[0]
    if np.mod(x.shape[0],2) == 1:
        x = x[:-1]
    print('here are some xs (turn starts and stops): ', x[0:10])

    turn_starts = x[0::2]
    turn_stops = x[1::2]
    print('turn_starts.shape,turn_stops,shape = ', turn_starts.shape,turn_stops.shape)

    left_starts = turn_starts[turns[turn_stops]== 1]

    right_starts = turn_starts[turns[turn_stops]== -1]

    labels = np.hstack([  np.ones(left_starts.shape[0]), -1*np.ones(right_starts.shape[0])         ])

    return labels,left_starts,right_starts


def make_timeseries_regressor(nn_params, nb_input_series=1, nb_outputs=1,custom_loss=0):
    model = Sequential()
    model.add(Conv1D(
       int(nn_params['nb_filter']*8), 
       kernel_size=int(nn_params['kernel']*8), 
       activation='relu', 
       input_shape=(nn_params['window'], nb_input_series)
    ))
    model.add(MaxPooling1D())
    model.add(Conv1D(
        int(nn_params['nb_filter']*4), 
        kernel_size=int(nn_params['kernel']*4), 
        activation='relu', 
        input_shape=(nn_params['window'], nb_input_series)
    ))
    model.add(MaxPooling1D())
    model.add(Conv1D(
        int(nn_params['nb_filter']*2),
        kernel_size=int(nn_params['kernel']*2), 
        activation='relu'
    ))
    model.add(MaxPooling1D())
    model.add(Conv1D(
        int(nn_params['nb_filter']), 
        kernel_size=int(nn_params['kernel']), 
        activation='relu'
    ))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(nn_params['nb_filter']*8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_outputs, activation='softmax')) ## was linear   # For binary classification, change the activation to 'sigmoid'
    
    adam = Adam(lr=nn_params['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # if custom_loss == 0:
    #     model.compile(loss='mae', optimizer=adam, metrics=['mse'])
    # else:
    #     model.compile(loss=modified_mse, optimizer=adam, metrics=['mse'])

    # To perform (binary) classification instead:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model

def evaluate_timeseries(timeseries1, timeseries2, nn_params,custom_loss=0):
    nb_samples, nb_series = timeseries1.shape
    
    if timeseries2.ndim == 1:
        timeseries2 = np.atleast_2d(timeseries2).T
    

    turn_labels,left_starts,right_starts = get_turn_idx(timeseries2)

    win_half = int(nn_params['window']/2)

    # left_mask = np.zeros(timeseries1.shape[0],dtype=bool)
    # right_mask = np.zeros(timeseries1.shape[0],dtype=bool)

    # for idx in range(len(left_starts)):
    #     left_mask[left_starts[idx]- win_half: left_starts[idx]+win_half ] = 1
    # for idx in range(len(right_starts)):
    #     right_mask[right_starts[idx]- win_half: right_starts[idx]+win_half ] = 1


    #left_X = timeseries1[left_mask,:]
    #right_X = timeseries1[right_mask,:]
    #X_turns = np.concatenate([left_X,right_X],axis=0)
    

    nb_out_samples, nb_out_series = timeseries2.shape

    print('THE SHAPES OF timeseries1, timeseries2 ==== ',timeseries1.shape, timeseries2.shape)    
    #X, y = make_timeseries_instances(timeseries1, timeseries2, nn_params['window'], nn_params['offset'])
    X = get_spikes_with_history(timeseries1,win_half,win_half)
    y = turn_labels
    #print('###################### getting non-zero values ######################')
    
    print(y.shape)
    print(X.shape)

    left_X = X[left_starts,:,:]
    right_X = X[right_starts,:,:]
    X_turns = np.concatenate([left_X,right_X],axis=0)
    print('X_turns.shape = ', X_turns.shape)
    print('turn_labels.shape = ', turn_labels.shape)
    
    # non_zeros = np.where(abs(y) > 0.25 )[0]    
    
    # y = y[non_zeros,:]
    # X = X[non_zeros,:,:]
    # pos = np.where(y > 0)[0]
    # neg = np.where(y < 0)[0]
    # y[neg] = -1
    # y[pos] = 1
    #y = Normalizer(norm='l2').fit_transform(np.atleast_2d(y))
    #y = (y - np.mean(y)) / np.std(y)
    
    print(y.shape)
    print(X.shape)

    #X, y = timeseries_shuffler(X, y, 3000, 25)
    
    if nn_params['verbose']: 
        print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series))
        print('\n\nExample input feature:', X[0], '\n\nExample output labels:', y[0])
    
    model = make_timeseries_regressor(
        nn_params,
        nb_input_series=nb_series, 
        nb_outputs=nb_out_series,
        custom_loss=custom_loss
    )
    
    if nn_params['verbose']: 
        print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape))
        model.summary()
    
    #X_train, X_test, y_train, y_test = split_data(X, y, 0.5)
    X_train, X_test, y_train, y_test = train_test_split(X_turns,turn_labels,test_size=0.5,random_state=42)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=5, 
        verbose=1, 
        mode='auto'
    )

    model.fit(
        X_train, 
        y_train, 
        epochs=nn_params['eps'], 
        batch_size=nn_params['bs'], 
        validation_data=(X_test, y_test) #,
        #callbacks=[early_stopping]
    )

    return model, X_train, X_test, y_train, y_test

def determine_class(X, y, y_key, nn_params, plot_result=True):

    if y_key[0].find('yaw') == -1:
        custom_loss = 0
        print('Training on %s, using MSE as loss function' % y_key[0])
    else:
        custom_loss = 1
        print('Training on %s, using custom loss function' % y_key[0])


    model, X_train, X_test, y_train, y_test = evaluate_timeseries(
        X, 
        y, 
        nn_params,
        custom_loss
    )
    
    y_test_hat = model.predict(X_test)
    
    R2s, rs = do_the_thing(
        y_test, 
        y_test_hat, 
        y_key, 
        'temp_conv_results_{}_y:{}'.format(nn_params['id'], y_key),
        plot_result=plot_result
    )
    
    return R2s, rs
