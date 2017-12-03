import numpy as np
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from data_helpers import pass_filter, split_data, make_timeseries_instances, timeseries_shuffler
from metrics_helper import do_the_thing
import keras.backend as K
from sklearn.preprocessing import Normalizer

def modified_mse(y_true, y_pred): #### modified MSE loss function for absolute yaw data (0-360 values wrap around)
    

    #y_true = y_true * y_std + y_mean ### y_train_mean,y_train_std are GLOBALS ??? 
    #y_pred = y_pred * y_std + y_mean

    mod_square =  K.square(K.abs(y_pred - y_true) - 360) ### hack 2.1 = (360 - np.mean(ox)) / np.std(ox) 2.1086953197291871
    raw_square =  K.square(y_pred - y_true)
    better = K.minimum(mod_square,raw_square)
    return K.mean(better,axis= -1)


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
        
    nb_out_samples, nb_out_series = timeseries2.shape
    print('THE SHAPES OF timeseries1, timeseries2 ==== ',timeseries1.shape, timeseries2.shape)    
    X, y = make_timeseries_instances(timeseries1, timeseries2, nn_params['window'], nn_params['offset'])

    print('###################### getting non-zero values ######################')
    non_zeros = np.where(abs(y) > 0.25 )[0]
    

    print(y.shape)
    print(X.shape)

    y = y[non_zeros,:]
    X = X[non_zeros,:,:]

    pos = np.where(y > 0)[0]
    neg = np.where(y < 0)[0]
    y[neg] = -1
    y[pos] = 1
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
    
    X_train, X_test, y_train, y_test = split_data(X, y, 0.5)
    
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

def determine_fit(X, y, y_key, nn_params, plot_result=True):

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
