import numpy as np
import pandas as pd
import sys,os
import pylab
from scipy import stats,signal,io
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn import linear_model
import pickle
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
import h5py
import matplotlib.gridspec as gridspec
plt.rcParams['pdf.fonttype'] = 'truetype'
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold


def get_all_files(input_file_path):
    
    all_files = []

    for file in os.listdir(input_file_path):
            if file.startswith("636"):
                all_files.append(file)


    return np.asarray(all_files)



if __name__ == "__main__":

    rat_types = ['V1','M2_lesion']
    rat_paths = {rat_types[0]: ['/n/coxfs01/guitchounts/ephys/%s/Analysis/'] * 5,
                 rat_types[1]: ['/n/coxfs01/guitchounts/ephys/%s/Analysis/']  * 4 }

    rats =  {rat_types[0]: ['GRat26','GRat27','GRat31','GRat36','GRat54'], rat_types[1]:['GRat47','GRat48','GRat49','GRat50']}

    time_vec = np.arange(-1.0,1.01,1/100.)

    exp_types = ['dark','light']
    dx_types = ['dx','dy','dz']
    npz_keys = ['y_left', 'y_right', 'X_left', 'X_right']
    dx_dict = {rat_type: 
               {dx_type : 
                {exp_type : 
                 {direction : 
                  [] for direction in npz_keys} for exp_type in exp_types } 
                for dx_type in dx_types } for rat_type in rat_types } # np.unique(exp_names)

    trial_sizes = {rat_type: 
               {dx_type : 
                {exp_type : 
                 {direction : 
                  [] for direction in npz_keys} for exp_type in exp_types } 
                for dx_type in dx_types } for rat_type in rat_types }

    for rat_type in rat_types:

        for rat_idx,rat in enumerate(rats[rat_type]):
            
            
            
            input_file_path = rat_paths[rat_type][rat_idx] % rat # os.getcwd() + '/../%s' % rat
            
            
            #rat = 'grat27'
            print(rat)
            exp_names = []
            all_files = get_all_files(input_file_path)
            for i,fil in enumerate(all_files):
                for exp in os.listdir(input_file_path + '/' + fil + '/'): ### e.g. ./636596531772835142
                    #print(exp)
                    if exp.startswith('%s_0' % rat.lower()):
                        exp_name = exp[exp.find('m_')+2:exp.find('.txt')]

                        exp_names.append(exp_name)
                    elif exp.startswith('%s_1' % rat.lower()):
                        exp_name = exp[exp.find('m_')+2:exp.find('.txt')]

                        exp_names.append(exp_name)

                #exp_names = np.asarray(exp_names)

                #for i,fil in enumerate(all_files):

                files_in_exp = os.listdir(input_file_path + '/%s/' % fil)
                #print(files_in_exp)

                for dx_type in dx_types:

                    dxs_in_exp = [s for s in files_in_exp if '%s_' % dx_type in s if '.npz' in s]  



                    for dx_in_exp in dxs_in_exp:
                        npz_path = input_file_path + '/%s/%s' % (fil,dx_in_exp)
                        dx = np.load(npz_path)

                        for key in dx.keys():
                            if len(dx[key].shape) == 2:
                                axis = (0)
                            elif len(dx[key].shape) == 3:
                                axis = (0,2)
                            #print(npz_path)
                            if exp_name in exp_types:
                                #print(npz_path)
                                print(rat,npz_path,exp_name,dx[key].shape)
                                dx_dict[rat_type][dx_type][exp_name][key].append(np.nanmean(dx[key],axis=axis))
                                trial_sizes[rat_type][dx_type][exp_name][key].append(dx[key].shape)

    #### Prep for Logistic Regression
    exp_lens = []

    label_names = []

    for rat_type in rat_types:
        for col,dx_type in enumerate(dx_types):
            for exp_type in exp_types:
                for npz_key in ['X_left', 'X_right'] :
                    exp_lens.append(len(dx_dict[rat_type][dx_type][exp_type][npz_key]))
                    label_names.append(str([rat_type,dx_type,exp_type,npz_key]))
                    print(dx_type,exp_type,npz_key,len(dx_dict[rat_type][dx_type][exp_type][npz_key]))

    min_trials = min(exp_lens)
    print('Min Trials = ', min_trials)

    turns_list = []

    for rat_type in rat_types:
        for col,dx_type in enumerate(dx_types):
            for exp_type in exp_types:
                for npz_key in ['X_left', 'X_right'] :  #npz_keys: #:

                    #plot_item = np.asarray([thing-np.mean(thing[0:50]) for thing in dx_dict[rat_type][dx_type][exp_type][npz_key]])
                    print(rat_type,dx_type,exp_type,npz_key,len(dx_dict[rat_type][dx_type][exp_type][npz_key]))
                    turns_list.append(np.vstack(dx_dict[rat_type][dx_type][exp_type][npz_key][0:min_trials]))
                    
    turns_list = np.asarray(turns_list)

    ### will first use the first 12 out of 24 to train/test (V1 not M2_lesion)
    ### those first 12 will get labels 0-11  x = 75:126
    num_labels = 12
    labels = np.empty([num_labels,min_trials])  # labels will be 12x84 [0,:] = 0 [1,:] = 1 etc
    
    for i in range(num_labels):
        labels[i,:] = np.repeat(i,min_trials)



    X_flat = np.reshape(turns_list[0:12,:,75:126],
                    [turns_list[0:12,:,75:126].shape[0] * turns_list[0:12,:,75:126].shape[1], turns_list[0:12,:,75:126].shape[2]])
                    ### 24 types, 84 observations, 51 samples. reshape this to num_observations x num_samples
    y = np.reshape(labels,[labels.shape[0]*labels.shape[1]  ])

    print('The shapes of X_flat and y are ', X_flat.shape,y.shape)

    X_test_M2_lesion = np.reshape(turns_list[12:,:,75:126],
                              [turns_list[12:,:,75:126].shape[0] * turns_list[12:,:,75:126].shape[1], turns_list[12:,:,75:126].shape[2]])

    y_test_M2_lesion = y ### the labels are the same!

    ###### Logistic Regression on varying numbers of trials
    random_state = 912883823
    train_test_splits = { }
    repeats = 100

    num_train_samples = []

    split_min,split_max = 2,20
    splits = range(split_min,split_max)
    splits.extend([30,40,50,60])

    for n_splits in splits:
        train_test_splits[n_splits] = []
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=int(repeats/2), random_state=random_state)

        #### !!!!! Reversing train and test here compared to normal usage (b/c want to train on the smaller number of trials)
        for test, train in rkf.split(range(X_flat.shape[0])):
             #print("%s %s" % (train, test))
            train_test_splits[n_splits].append([train,test])

        num_train_samples.append(len(train))


    parameters = {'penalty':["l2"],'C':np.logspace(-3,3,4)  } # np.logspace(-3,3,4)  # [1.0]

    
    clfs = {}
    m2_clfs = {}
    predictions = {}    
    predictions_m2_v1train = {}

    clfs_scores = {}
    clfs_scores_m2 = {}
    clfs_scores_m2_v1train = {}

    shuffle_scores = {}
    shuffle_scores_m2 = {}

    for n_splits in splits:
            
        clfs[n_splits] = []
        m2_clfs[n_splits] = []
        predictions[n_splits] = []
        predictions_m2_v1train[n_splits] = []
        clfs_scores[n_splits] = []
        clfs_scores_m2[n_splits] = []
        
        clfs_scores_m2_v1train[n_splits] = []
        
        shuffle_scores[n_splits] = []
        shuffle_scores_m2[n_splits] = []
        

        for i in range(repeats): #### The number of splits goes up as you train on fewer trials, but we'll stick to the first 100 splits

            
            X_train = X_flat[train_test_splits[n_splits][i][0],:]
            y_train = y[train_test_splits[n_splits][i][0]]

            X_test = X_flat[train_test_splits[n_splits][i][1],:]
            y_test = y[train_test_splits[n_splits][i][1]]

            ##### X_test_M2_lesion = full set of M2 trials. = 984 x 51 
            X_train_m2 = X_test_M2_lesion[train_test_splits[n_splits][i][0],:]
            y_train_m2 = y_test_M2_lesion[train_test_splits[n_splits][i][0]]
            
            X_test_m2 = X_test_M2_lesion[train_test_splits[n_splits][i][1],:]
            y_test_m2 = y_test_M2_lesion[train_test_splits[n_splits][i][1]]
            
            
            log_reg = LogisticRegression() ## multi_class='ovr'
            clf = GridSearchCV(log_reg, parameters,scoring='accuracy',n_jobs=4) # 
            clf.fit(X_train, y_train)
            clfs[n_splits].append(clf)
            predictions[n_splits].append(clf.predict(X_test))
            predictions_m2_v1train[n_splits].append(clf.predict(X_test_M2_lesion))
            
            clfs_scores[n_splits].append(clf.score(X_test,y_test))
            clfs_scores_m2_v1train[n_splits].append(clf.score(X_test_M2_lesion,y_test_M2_lesion))
            shuffle_scores[n_splits].append(clf.score(np.random.permutation(X_test),y_test))
            
            
            clf = None
            
            #### same for the M2-trained model:
            
            log_reg_m2 = LogisticRegression() ## multi_class='ovr'
            clf_m2 = GridSearchCV(log_reg_m2, parameters,scoring='accuracy',n_jobs=4)
            clf_m2.fit(X_train_m2, y_train_m2)
            m2_clfs[n_splits].append(clf_m2)
            clfs_scores_m2[n_splits].append(clf_m2.score(X_test_m2,y_test_m2))
            shuffle_scores_m2[n_splits].append(clf_m2.score(np.random.permutation(X_test_m2),y_test_m2))
            
            
            print('Repeat #%d. Score = %f, M2 Score = %f, M2 V1-train Score = %f, Shuffle Score = %f, M2 Shuffle Score = %f' 
                  % (i,clfs_scores[n_splits][i],clfs_scores_m2[n_splits][i], 
                     clfs_scores_m2_v1train[n_splits][i],shuffle_scores[n_splits][i],shuffle_scores_m2[n_splits][i]  ) ) 
            
            clf_m2 = None

    #### Save stuff!

        np.savez('scores_%d.npz' % n_splits, scores=clfs_scores[n_splits])

        np.savez('scores_m2_%d.npz' % n_splits, scores=clfs_scores_m2[n_splits])

        np.savez('scores_m2_v1train_%d.npz' % n_splits, scores=clfs_scores_m2_v1train[n_splits])

        np.savez('shuffle_scores_m2_%d.npz' % n_splits, scores=shuffle_scores_m2[n_splits])

        np.savez('shuffle_scores_%d.npz' % n_splits, scores=shuffle_scores[n_splits])

        
    ### plot stuff!!

    f = plt.figure(dpi=600)
    labels = ['V1 Scores','M2 Scores','M2 Scores V1-train','Shuffle Scores','Shuffle Scores M2']

    for i,plot_dict in enumerate([clfs_scores,clfs_scores_m2,clfs_scores_m2_v1train,shuffle_scores,shuffle_scores_m2]):

        plot_means = [np.mean(plot_dict[n_splits]) for n_splits in splits]
        plot_sems = [stats.sem(plot_dict[n_splits]) for n_splits in splits]

        plt.errorbar(num_train_samples[::-1],plot_means[::-1],yerr=plot_sems[::-1],label=labels[i])


    plt.ylabel('Accuracy')
    plt.xlabel('Training ')
    plt.legend()
    sns.despine(offset=10)

    f.savefig('logistic_results.pdf')




