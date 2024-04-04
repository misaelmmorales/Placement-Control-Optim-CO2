########################################################################################################################
################################ SPATIAL DATA ASSIMILATION IN GEOLOGIC CO2 SEQUESTRATION ###############################
########################################################################################################################
import os, sys, shutil
from time import time
from tqdm import tqdm
from units import *

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd as SVD
from numpy.matlib import repmat
from scipy.io import loadmat, savemat
from pyesmda import ESMDA, approximate_cov_mm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import mean_squared_error as img_mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_mutual_information as img_nmi

octave_cli_path = 'C:/Users/381792/AppData/Local/Programs/GNU Octave/Octave-8.2.0/mingw64/bin/octave-cli.exe'
#octave_cli_path = 'C:/Program Files/GNU Octave/Octave-8.2.0/mingw64/bin/octave-cli-8.2.0'
os.environ['OCTAVE_EXECUTABLE'] = octave_cli_path
import oct2py
from oct2py import Oct2Py
from oct2py import Struct, Cell

class spatialDA:
    def __init__(self):
        self.return_data  = True
        self.return_plot  = True
        self.save_data    = False
        self.verbose      = True
        
        self.dim          = 51
        self.n_ensemble   = 100
        self.years        = [1,3,5]
        self.mrst_address = 'C:/Users/381792/Documents/mrst-2023a/'
        #self.mrst_address = 'E:/mrst-2023a'

    def check_torch_gpu(self):
        '''
        Check torch build in python to ensure GPU is available for training.
        '''
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        py_version, conda_env_name = sys.version, sys.executable.split('\\')[-2]
        print('-------------------------------------------------')
        print('------------------ VERSION INFO -----------------')
        print('Conda Environment: {} | Python version: {}'.format(conda_env_name, py_version))
        print('Torch version: {}'.format(torch_version))
        print('Torch build with CUDA? {}'.format(cuda_avail))
        print('# Device(s) available: {}, Name(s): {}\n'.format(count, name))
        self.device = torch.device('cuda' if cuda_avail else 'cpu')
        return None
    
    def mrst_startup(self):
        self.oc = Oct2Py()
        self.oc.warning('off')
        self.oc.addpath(self.oc.genpath(self.mrst_address))
        self.oc.eval('startup')
        if self.return_data:
            return self.oc       
        
    def load_perm_sat_true(self):
        self.perm_true = np.log10(np.array(pd.read_csv('Syn_GroundTruth_Linux/PERMX_true_syn.inc'))).squeeze()
        self.sat_true  = np.load('simulations/sat_true.npy').reshape(3,self.dim*self.dim)
        if self.save_data:
            np.save('simulations/perm_true.npy', self.perm_true)
            np.save('simulations/sat_true.npy', self.sat_true)
        if self.verbose:
            print('True Perm: {} | True Saturation: {}'.format(self.perm_true.shape, self.sat_true.shape))
        if self.return_data:
            return self.perm_true, self.sat_true
            
    def load_perm_sat_ens(self):
        self.perm_ens = np.zeros((self.n_ensemble, self.dim*self.dim))
        for i in range(self.n_ensemble):
            self.perm_ens[i] = np.log10(np.array(pd.read_csv('1n3n5/ensemble/3DMODEL/PERMX_{}.inc'.format(i+1)))).squeeze()
            self.sat_ens     = loadmat('simulations/sat_ens.mat')['sat']
        if self.save_data:
            np.save('simulations/perm_ens.npy', self.perm_ens)
            savemat('simulations/sats_ens.mat', {'sat':self.sat_ens})
        if self.verbose:
            print('Perm Ensemble: {} | Saturation Ensemble: {}'.format(self.perm_ens.shape, self.sat_ens.shape))
        if self.return_data:
            return self.perm_ens, self.sat_ens
        
    def load_perm_all(self):
        self.perm_all = np.zeros((self.n_ensemble+1, self.dim*self.dim))
        for i in range(self.n_ensemble+1):
            self.perm_all[i] = loadmat('simulations/perm_ensemble/perm_{}.mat'.format(i))['perm'].reshape(self.dim*self.dim)
        if self.save_data:
            np.save('simulations/perm_all.npy', self.perm_all)
        if self.verbose:
            print('Perm All shape: {}'.format(self.perm_all.shape))
        if self.return_data:
            return self.perm_all

    def read_file(self, filename):
        sat_data, self.sat_map = np.zeros(self.dim*self.dim), np.zeros((self.dim,self.dim))
        count = 0
        for lines in open(filename):
            count += 1
            lines = lines.rstrip('\n')
            count_col = 0
            if count>26011:
                for e in lines.split():
                    count_col += 1
                    if count_col == 5:
                        sat_data[count-26012] = float(e)
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                self.sat_map[i,j] = sat_data[j+i*self.dim]
        if self.return_data:
            return self.sat_map
    
    def plot_perm_sat(self, perm, sat, mask=True, permtitle='Perm', figsize=(25,5), interp=['hanning','none'], cmaps=['jet','jet']):
        fig, axs = plt.subplots(1,4,figsize=figsize)
        ticks, ticklabs = np.linspace(0,50,num=5), np.linspace(0,4000,num=5,dtype='int')
        imk = axs[0].imshow(perm.reshape(self.dim,self.dim), cmap=cmaps[0], interpolation=interp[0])      
        plt.colorbar(imk, fraction=0.046, pad=0.04)
        axs[0].set(title=permtitle, xlabel='X [m]', ylabel='Y [m]', xticks=ticks, yticks=ticks, xticklabels=ticklabs, yticklabels=ticklabs)
        for i in range(1,4):
            s = sat[i-1].reshape(self.dim,self.dim)
            sm = np.ma.masked_where(s==0,s)
            if mask:
                ims = axs[i].imshow(sm, cmap=cmaps[1], interpolation=interp[-1])
            else:
                ims = axs[i].imshow(s, cmap=cmaps[1], interpolation=interp[-1])
            plt.colorbar(ims, fraction=0.046, pad=0.04)
            axs[i].set(title='Year {}'.format(self.years[i-1]), xlabel='X [m]', xticks=ticks, xticklabels=ticklabs, yticks=[])
        plt.show()
        
    def plot_ens_mean_std(self, titles=['Ensemble Mean','Ensemble Standard Deviation'], figsize=(15,3)):
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ticks, ticklabs = np.linspace(0,50,num=5), np.linspace(0,4000,num=5,dtype='int')
        im1 = axs[0].imshow(self.perm_ens.mean(0).reshape(51,51), cmap='jet', interpolation='gaussian')
        im2 = axs[1].imshow(self.perm_ens.std(0).reshape(51,51), cmap='jet', interpolation='gaussian')
        for i in range(2):
            axs[i].set(xlabel='X [m]', xticks=ticks, xticklabels=ticklabs, 
                    ylabel='Y [m]', yticks=ticks, yticklabels=ticklabs,
                    title=titles[i])
        plt.colorbar(im1, pad=0.04, fraction=0.046, label='$\mu_k$ [logmD]')
        plt.colorbar(im2, pad=0.04, fraction=0.046, label='$\sigma^2(k)$ [logmD]')
        plt.show()