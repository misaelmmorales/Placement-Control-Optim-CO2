import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt

NUM_REALIZATIONS = 1000
NX,  NY,  NZ = 64, 64, 1
NTT, NT1, NT2 = 40, 20, 5

sec2year   = 365.25 * 24 * 60 * 60
Darcy      = 9.869233e-13
psi2pascal = 6894.76
co2_rho    = 686.5266
milli      = 1e-3
mega       = 1e6

deltatime = sio.loadmat('data/time_arr.mat', simplify_cells=True)['time_arr']
timesteps = np.cumsum(deltatime)
timesteps_inj = timesteps[:20]
timesteps_mon = timesteps[[19, 24, 29, 34, 39]]

print('timesteps: {} | deltatime: {}'.format(len(timesteps), np.unique(deltatime)))
print('injection: {}'.format(timesteps_inj))
print('monitoring: {}'.format(timesteps_mon))

gt = sio.loadmat('grids/Gt.mat', simplify_cells=True)['Gt']
gtops = gt['cells']['z'].reshape(NX,NY,order='F')
gvols = gt['parent']['cells']['volumes'].reshape(NX,NY,order='F')
ghght = gt['cells']['H'].reshape(NX,NY,order='F')

poro = np.zeros((NUM_REALIZATIONS, NX, NY))
perm = np.zeros((NUM_REALIZATIONS, NX, NY))
well = np.zeros((NUM_REALIZATIONS, NX, NY))
ctrl = np.zeros((NUM_REALIZATIONS, 5, 20))

for i in tqdm(range(NUM_REALIZATIONS)):
    r = sio.loadmat('rock/rock_{}.mat'.format(i), simplify_cells=True)['var']
    poro[i] = r['poro'].reshape(NX,NY,order='F')
    perm[i] = np.log10(r['perm'].reshape(NX,NY,order='F') / milli / Darcy)

    ww = np.zeros((64,64))
    w = sio.loadmat('well_locs/well_locs_{}.mat'.format(i), simplify_cells=True)['var'] - 1
    if len(w.shape)==1:
        w = w.reshape(1,2)
    well[i, w[:,0], w[:,1]] = 1

    c = sio.loadmat('controls/controls_{}.mat'.format(i), simplify_cells=True)['var']
    if len(c.shape)==1:
        c = c.reshape(1,-1)
    nc = c.shape[0]
    ctrl[i,:nc] = c
c_data = np.moveaxis(ctrl, -1, 1)

tops = np.repeat(np.expand_dims(gtops, axis=0), NUM_REALIZATIONS, axis=0)
hght = np.repeat(np.expand_dims(ghght, axis=0), NUM_REALIZATIONS, axis=0)
X_data = np.stack([poro,perm,well,tops,hght], axis=-1)
print('X_data: {} | c_data: {}'.format(X_data.shape, c_data.shape))

pressure = np.zeros((NUM_REALIZATIONS, NTT, NX, NY))
saturation = np.zeros((NUM_REALIZATIONS, NTT, NX, NY))

for i in tqdm(range(NUM_REALIZATIONS)):
    d = sio.loadmat('states/states_{}.mat'.format(i), simplify_cells=True)['var']
    for t in range(NTT):
        pressure[i,t] = d[t]['pressure'].reshape(NX,NY,order='F') / psi2pascal
        saturation[i,t] = d[t]['s'].reshape(NX,NY,order='F')

y1_data = np.stack([pressure[:,:NT1], saturation[:,:NT1]], axis=-1)
y2_data = np.expand_dims(saturation[:,NT1:], -1) #[:,[0,4,9,14,19]]
print('y1_data: {} | y2_data: {}'.format(y1_data.shape, y2_data.shape))

np.save('data/X_data.npy', X_data)
np.save('data/c_data.npy', c_data)
np.save('data/y1_data.npy', y1_data)
np.save('data/y2_data.npy', y2_data)

print('... Done!')