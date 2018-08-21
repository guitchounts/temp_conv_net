import numpy as np
# from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
from data_helpers import pass_filter, split_data, make_timeseries_instances, timeseries_shuffler,sample_dx_uniformly
from metrics_helper import do_the_thing
# import keras.backend as K
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from sklearn.externals import joblib

def modified_mse(y_true, y_pred): #### modified MSE loss function for absolute yaw data (0-360 values wrap around)
    

    #y_true = y_true * y_std + y_mean ### y_train_mean,y_train_std are GLOBALS ??? 
    #y_pred = y_pred * y_std + y_mean

    mod_square =  K.square(K.abs(y_pred - y_true) - 360) ### hack 2.1 = (360 - np.mean(ox)) / np.std(ox) 2.1086953197291871
    raw_square =  K.square(y_pred - y_true)
    better = K.minimum(mod_square,raw_square)
    return K.mean(better,axis= -1)

def get_turn_idx(dx):

    turns = np.zeros(dx.shape)

    turns[left > 1] = 1

    turns[right > 1] = -1

    x =np.where(np.diff(turns) )[0]

    turn_starts = x[0::2]
    turn_stops = x[1::2]

    left_starts = turn_starts[turns[turn_stops]==1]

    right_starts = turn_starts[turns[turn_stops]==-1]

    return left_starts,right_starts

# def make_ridgeCV_model():
 
    
#     print('********************************** Making RidgeCV Model **********************************')
#     #Declare model
#     model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],normalize=True,fit_intercept=True)
  
#     return model

def make_linear_model(model_type='ridge'):
 
    if model_type == 'ridge':
        print('********************************** Making RidgeCV Model **********************************')
        #Declare model
        model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0],normalize=True,fit_intercept=True)
    elif model_type == 'lasso':

        print('********************************** Making LassoCV Model **********************************')
        #Declare model
        model = linear_model.LassoCV(n_alphas =5,normalize=True,fit_intercept=True,max_iter=100000,n_jobs=-1) #
  
    return model


def make_timeseries_regressor(nn_params, nb_input_series=1, nb_outputs=1,custom_loss=0):
    model = Sequential()
    # model.add(Conv1D(
    #    int(nn_params['nb_filter']*8), 
    #    kernel_size=int(nn_params['kernel']*8), 
    #    activation='relu', 
    #    input_shape=(nn_params['window'], nb_input_series)
    # ))
    #model.add(MaxPooling1D())
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
    model.add(Dense(nb_outputs, activation='linear')) ## was linear   # For binary classification, change the activation to 'sigmoid'
    
    adam = Adam(lr=nn_params['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    if custom_loss == 0:
        model.compile(loss='mse', optimizer=adam, metrics=['mse'])
    else:
        model.compile(loss=modified_mse, optimizer=adam, metrics=['mse'])

    # To perform (binary) classification instead:
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model

def evaluate_timeseries(timeseries1, timeseries2, nn_params,custom_loss=0,model_type = 'temp_conv'):
    nb_samples, nb_series = timeseries1.shape
    
    if timeseries2.ndim == 1:
        timeseries2 = np.atleast_2d(timeseries2).T
    


    if 'shuffle' in nn_params.keys():
        shuffle = nn_params['shuffle'] ## if the params file has a shuffle key, use it; otherwise set it to False 
    else:
        shuffle = False


    nb_out_samples, nb_out_series = timeseries2.shape

    print('THE SHAPES OF timeseries1, timeseries2 ==== ',timeseries1.shape, timeseries2.shape)    
    X, y = make_timeseries_instances(timeseries1, timeseries2, nn_params['window'], nn_params['offset'])
    print('Shapes of X and y after making timeseries instance:', X.shape,y.shape)


    if 'resample' in nn_params.keys():
        resample = nn_params['resample']
    else:
        resample = 0
        
    if resample == 1:
        print('###################### resampling y ######################')

        sampled_dx_idx = sample_dx_uniformly(y)
        sampled_dx_idx = np.sort(sampled_dx_idx)
        
        y = y[sampled_dx_idx,:]
        X = X[sampled_dx_idx,:,:]
        print('Shapes of X and y after resampling:', X.shape,y.shape)
    
    print(y.shape)
    print(X.shape)


    # non_zeros = np.where(abs(y) > 0.1 )[0]    
    
    # y = y[non_zeros,:]
    # X = X[non_zeros,:,:]
    # pos = np.where(y > 0)[0]
    # neg = np.where(y < 0)[0]
    # y[neg] = -1
    # y[pos] = 1
    #y = Normalizer(norm='l2').fit_transform(np.atleast_2d(y))
    #y = (y - np.mean(y)) / np.std(y)
    
    

    X, y = timeseries_shuffler(X, y, 3000, 25)
    
    if nn_params['verbose']: 
        print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series))
        print('\n\nExample input feature:', X[0], '\n\nExample output labels:', y[0])
    
    if model_type == 'ridge' or model_type == 'lasso':
        print(model_type)
        print('Making Linear %s Model' % model_type)
        model = make_linear_model(model_type=model_type)
        
    else:
        print(model_type)
        print('Making TempConvNet Model')
        model = make_timeseries_regressor(
            nn_params,
            nb_input_series=nb_series, 
            nb_outputs=nb_out_series,
            custom_loss=custom_loss
        )
    

    if nn_params['verbose']: 
        print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape))
        model.summary()
    
    X_train, X_test, y_train, y_test = split_data(X, y, 0.5,shuffle=shuffle)
    
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    # print('###################### resampling y ######################')

    # sampled_dx_idx_train = sample_dx_uniformly(y_train,num_points=5000)
    # sampled_dx_idx_test = sample_dx_uniformly(y_test,num_points=5000)

    # y_train = y_train[sampled_dx_idx_train,:]
    # X_train = X_train[sampled_dx_idx_train,:,:]

    # y_test = y_test[sampled_dx_idx_test,:]
    # X_test = X_test[sampled_dx_idx_test,:,:]
    # print('Shapes of X_train and y_train after resampling:', X_train.shape,y_train.shape)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=20, 
        verbose=1, 
        mode='auto'
    )

    if model_type == 'ridge' or model_type == 'lasso':
        print('Reshaping X_train and X_test and fitting %s model' % model_type)
        #### X's are (time, window, channels), e.g. (13085, 200, 16). Reshape for the linear model:
        X_train = X_train.reshape(X_train.shape[0],(X_train.shape[1]*X_train.shape[2]))
        X_test = X_test.reshape(X_test.shape[0],(X_test.shape[1]*X_test.shape[2]))
        model.fit(X_train,y_train)

        

    else:
        model.fit(
            X_train, 
            y_train, 
            epochs=nn_params['eps'], 
            batch_size=nn_params['bs'], 
            validation_data=(X_test, y_test),
            callbacks=[early_stopping]
        )

    return model, X_train, X_test, y_train, y_test

def determine_fit(X, y, y_key, nn_params,save_dir, plot_result=True,model_type = 'temp_conv'):

    # if y_key[0].find('yaw') == -1:
    custom_loss = 0
    #     print('Training on %s, using MSE as loss function' % y_key[0])
    # else:
    #     custom_loss = 1
    #     print('Training on %s, using custom loss function' % y_key[0])


    model, X_train, X_test, y_train, y_test = evaluate_timeseries(
        X, 
        y, 
        nn_params,
        custom_loss,
        model_type=model_type
    )
    
    y_test_hat = model.predict(X_test)
    
    if model_type == 'ridge' or model_type == 'lasso':
        # save the model:
        joblib.dump(model, save_dir + str(y_key) + '_%s.pkl' % model_type) 

    R2s, rs = do_the_thing(
        y_test, 
        y_test_hat, 
        y_key, 
        'temp_conv_results_{}_{}'.format(nn_params['id'], y_key),
        save_dir,
        plot_result=plot_result
    )
    
    return R2s, rs
