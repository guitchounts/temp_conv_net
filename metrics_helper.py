import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('white')
mpl.rcParams['agg.path.chunksize'] = 100000

def get_R2(y_valid,y_hat):
    y_mean=np.mean(y_valid)
    R2=1-np.sum((y_hat-y_valid)**2)/np.sum((y_valid-y_mean)**2)
    return R2

def analyze_results(y_valids, y_hats):
    assert y_valids.shape == y_hats.shape
    R2s = [get_R2(y_valids, y_hats)] #### was R2s = [get_R2(y_valids[:,i], y_hats[:,i]) for i in range(y_valids.shape[1])]
    rs = [pearsonr(y_valids, y_hats)[0]]
    return R2s, rs

def plot_results(y_valids, y_hats, y_names, R2s, rs,save_dir, model_name='GRU'):
    num_figs = len(y_names)
    f = plt.figure(dpi=600,figsize=(7,3))
    f.suptitle(model_name, fontsize=10)
    
    gs = gridspec.GridSpec(num_figs, 7)

    for i in range(num_figs):
        y_valid = y_valids[:,i]
        y_valid_predicted = y_hats[:,i]

        ax1 = plt.subplot(gs[i, 0:4])
        ax2 = plt.subplot(gs[i, 4])
        ax3 = plt.subplot(gs[i, 5:])

        axarr = [ax1,ax2,ax3]

        axarr[0].plot(y_valid,linewidth=0.2,c='black')
        axarr[0].set_ylabel(y_names[i])
        axarr[0].plot(y_valid_predicted,linewidth=0.2,color='red')
        axarr[0].set_title('R^2 = %f. r = %f' % (R2s[i],rs[i]),fontsize= 12)

        axarr[1].scatter(y_valid,y_valid_predicted,alpha=0.05,s=2,marker='o')

        axarr[1].axis('equal')
        axarr[1].axes.xaxis.set_ticklabels([])
        axarr[1].axes.yaxis.set_ticklabels([])

        axarr[2].hist(y_valid,bins=100,color='black',alpha=.5)
        axarr[2].hist(y_valid_predicted,bins=100,color='red',alpha=.5)
        axarr[2].axes.xaxis.set_ticklabels([])
        axarr[2].axes.yaxis.set_ticklabels([])

        if i == num_figs-1:
            axarr[0].set_xlabel('Time (samples)')
            axarr[1].set_xlabel('Actual')
            axarr[1].set_ylabel('Predicted')
        else:
            axarr[0].axes.xaxis.set_ticklabels([])

    sns.despine(left=True,bottom=True)

    plt.tight_layout()

    f.savefig(save_dir + model_name + '.png')
    return plt

def do_the_thing(y_valids, y_hats, y_names, model_name,save_dir, plot_result=False):
    R2s, rs = analyze_results(y_valids, y_hats)
    print('******************************** saving results! ********************************')
    np.savez(save_dir + str(y_names) + '_results.npz',y_valids=y_valids,y_hats=y_hats)
    if plot_result:
        plot_results(y_valids, y_hats, y_names, R2s, rs,save_dir, model_name=model_name)
    
    return R2s, rs
