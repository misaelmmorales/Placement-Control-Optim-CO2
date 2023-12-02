import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.util import random_noise

import keras.backend as K
from keras import Model, regularizers
from keras.layers import *
from keras.optimizers import SGD, Adam, Nadam
from keras.losses import MeanSquaredError, MeanAbsoluteError

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow_addons.layers import *
from tensorflow_addons.optimizers import AdamW
from tensorflow.image import ssim as SSIM
from tensorflow.keras.metrics import mean_squared_error as MSE
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    cuda_version, cudnn_version = sys_info['cuda_version'], sys_info['cudnn_version']
    num_gpu_avail = len(tf.config.experimental.list_physical_devices('GPU'))
    gpu_name = device_lib.list_local_devices()[1].physical_device_desc[17:40]
    print('... Checking Tensorflow Version ...')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print("TF: {} | CUDA: {} | cuDNN: {}".format(tf.__version__, cuda_version, cudnn_version))
    print('# GPU available: {} ({})'.format(num_gpu_avail, gpu_name))
    #print(tf.config.list_physical_devices())
    return None

#################### DATALOADER ####################
def load_data(nx=32, ny=32, nt=60, nR=100, save=False, verbose=True):
    logperm    = np.array(pd.read_csv('simulations/perm_realization.csv')).reshape(nx,ny)
    perm       = loadmat('data_mat/permeability.mat')['permeability'].reshape(nx,ny,3)
    poro       = loadmat('data_mat/porosity.mat')['poro'].reshape(nx,ny)
    timesteps  = loadmat('data_mat/time_yr.mat')['cum_time']
    bhp        = loadmat('data_mat/bhp.mat')['bhp']
    pressure   = loadmat('data_mat/pressure.mat')['pressure'].reshape(nR,nx,ny,nt)
    saturation = loadmat('data_mat/saturation.mat')['saturation'].reshape(nR,nx,ny,nt)
    well_locs  = loadmat('data_mat/well_locations.mat')['well_locations']
    well_locs_mask = np.zeros((nR,nx,ny))
    for i in range(nR):
        well_locs_mask[i, well_locs[i][0], well_locs[i][1]] = 1  
    if save:
        np.save('data_npy/timesteps.npy', timesteps); np.save('data_npy/bhp.npy', bhp)
        np.save('data_npy/well_locs.npy', well_locs); np.save('data_npy/well_locs_mask.npy', well_locs_mask)
        np.save('data_npy/poro.npy', poro);           np.save('data_npy/perm.npy', perm)
        np.save('data_npy/pressure.npy', pressure);   np.save('data_npy/saturation.npy', saturation)
    if verbose:
        print('Log-Perm: {}'.format(logperm.shape))
        print('Permeability: {} | Porosity: {}'.format(perm.shape, poro.shape))
        print('Pressure: {} | Saturation: {}'.format(pressure.shape, saturation.shape))
        print('Timesteps: {} | BHP: {}'.format(timesteps.shape, bhp.shape))
        print('Well Locations: {} | Well Locations MASK: {}'.format(well_locs.shape, well_locs_mask.shape))
    return logperm, perm, poro, timesteps, bhp, pressure, saturation, well_locs, well_locs_mask

#################### PLOTS ####################
def plot_well_locs(wdata, kdata, figsize=(4,4), color='k', cmap='jet', title='Random Well Locations', xlab='X index', ylab='Y index', cbar_label='$log(k_x)$ [log-mD]'):
    plt.figure(figsize=figsize)
    plt.scatter(wdata[:,0], wdata[:,1], c=color)
    im = plt.imshow(kdata, cmap=cmap, aspect='auto')
    plt.colorbar(im, label=cbar_label, fraction=0.046, pad=0.04); 
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.title(title)
    
def plot_bhps(bhpdata, timedata, realization, figsize=(6,4), title='BHP vs. Time', xlab='Time [yrs]', ylab='BHP [psia]'):
    plt.figure(figsize=figsize)
    for k in realization:
        plt.plot(timedata, bhpdata[k], label='Realization {}'.format(k))
    plt.title(title); plt.xlabel(xlab); plt.ylabel(ylab)
    plt.legend(); plt.grid('on')
    
def plot_static(poro, perm, well_loc, ncols, multiplier=1, cmaps=['jet','jet'], figsize=(15,5)):
    logpermx = np.log10(perm[:,:,0])
    fig, axs = plt.subplots(2, ncols, figsize=figsize, facecolor='white')
    for j in range(ncols):
        k = j*multiplier
        im0 = axs[0,j].imshow(poro, cmap=cmaps[0])
        im1 = axs[1,j].imshow(logpermx, cmap=cmaps[1])
        axs[0,j].set(title='Realization {}'.format(k))
        for i in range(2):
            axs[i,j].scatter(well_loc[k][0], well_loc[k][1], c='k', marker='o')
            axs[i,j].set(xticks=[], yticks=[])
    axs[0,0].set(ylabel='Porosity'); axs[1,0].set(ylabel='LogPerm_x') 
    plt.colorbar(im0, label='$\phi$ [v/v]', fraction=0.046, pad=0.04)
    plt.colorbar(im1, label='$log(k_x)$ [log-mD]', fraction=0.046, pad=0.04)

def plot_dynamic(ddata, sdata, well_loc, nrows, multiplier=1, cmap='jet', figsize=(18,4), dtitle='Dynamic', stitle='Static'):
    k, j_timesteps = 0, np.insert(np.linspace(0, 60, 13, dtype='int')[1:]-1, 0, 0)
    fig, axs = plt.subplots(nrows, len(j_timesteps)+1, figsize=figsize)
    plt.suptitle(dtitle)
    for i in range(nrows):
        ims=axs[i,0].imshow(sdata, cmap='jet'); axs[i,0].set(xticks=[], yticks=[]); axs[0,0].set_title(stitle)
        for j in range(len(j_timesteps)):
            imd=axs[i,j+1].imshow(ddata[k,:,:,j_timesteps[j]], cmap=cmap)
            axs[0,j+1].set(title='t={}'.format(j_timesteps[j]+1))
            axs[i,0].set(ylabel='n={}'.format(k))
            axs[i,j+1].set(xticks=[], yticks=[])
        for j in range(len(j_timesteps)+1):
            axs[i,j].scatter(well_loc[k][0], well_loc[k][1], c='k', marker='o')
        k += multiplier