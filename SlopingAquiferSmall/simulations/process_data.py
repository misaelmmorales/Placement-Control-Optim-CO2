import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm

NX = 64
NY = 64
NTT = 40
NT0 = 20
psi2pascal = 6894.76

rnames = os.listdir('simulations/rock')
vnames = os.listdir('simulations/porevol')
wnames = os.listdir('simulations/well_locs')
cnames = os.listdir('simulations/controls')
snames = os.listdir('simulations/states')
tops = sio.loadmat('simulations/grids/G.mat', simplify_cells=True)['G']['cells']['centroids'].reshape(NX,NY,3,order='F')[...,-1]
heights = sio.loadmat('simulations/grids/Gt.mat', simplify_cells=True)['Gt']['cells']['H'].reshape(NX,NY,order='F')
gv = sio.loadmat('simulations/grids/G.mat', simplify_cells=True)['G']['cells']['volumes'].reshape(64,64,order='F')

poro = np.zeros((len(rnames), NX, NY))
perm = np.zeros((len(rnames), NX, NY))
#pvol = np.zeros((len(vnames), NX, NY))
wloc = np.zeros((len(wnames), NX, NY))
gtop = np.repeat(np.expand_dims(tops, axis=0), len(rnames), axis=0)
ghts = np.repeat(np.expand_dims(heights, axis=0), len(rnames), axis=0)
gvol = np.repeat(np.expand_dims(gv, axis=0), len(rnames), axis=0)
controls = np.zeros((len(cnames),20,5))
pressure = np.zeros((len(snames), NTT, NX, NY))
saturation = np.zeros((len(snames), NTT, NX, NY))

for i, n in tqdm(enumerate(rnames), desc='Loading Rock'):
    r = sio.loadmat('simulations/rock/{}'.format(n), simplify_cells=True)['var']
    poro[i] = r['poro'].reshape(NX,NY,order='F')
    perm[i] = np.log10(r['perm']).reshape(NX,NY,order='F')

# for i, n in tqdm(enumerate(vnames), desc='Loading PoreVol'):
#     pvol[i] = sio.loadmat('simulations/porevol/{}'.format(n), simplify_cells=True)['var'].reshape(NX,NY,order='F')

for i, n in tqdm(enumerate(wnames), desc='Loading Well Locs'):
    w = sio.loadmat('simulations/well_locs/{}'.format(n), simplify_cells=True)['var'] - 1
    if len(w.shape) == 1:
        w = w.reshape(1,-1)
    wloc[i, w[:,0], w[:,1]] = 1

for i, n in tqdm(enumerate(cnames), desc='Loading Controls'):
    c = sio.loadmat('simulations/controls/{}'.format(n), simplify_cells=True)['var']
    if len(c.shape) == 1:
        c = c.reshape(1,-1)
    nc = c.shape[0]
    controls[i,:,:nc] = c.T

for i, n in tqdm(enumerate(snames), desc='Loading States'):
    s = sio.loadmat('simulations/states/{}'.format(n), simplify_cells=True)['var']
    for j in range(NTT):
        pressure[i,j] = s[j]['pressure'].reshape(NX,NY,order='F') / psi2pascal
        saturation[i,j] = s[j]['s'][:,1].reshape(NX,NY,order='F')

X_data = np.stack([poro, perm, wloc, gtop, gvol], axis=-1) #pvol, ghts
c_data = controls
y_data = np.stack([pressure, saturation], axis=-1)
y1_data = y_data[:,:NT0]
y2_data = np.expand_dims(y_data[:,NT0:,...,-1], -1)
print('X: {} | c: {} | y1: {} | y2: {}'.format(X_data.shape, c_data.shape, y1_data.shape, y2_data.shape))

np.save('simulations/data/X_data.npy', X_data)
np.save('simulations/data/c_data.npy', c_data)
np.save('simulations/data/y1_data.npy', y1_data)
np.save('simulations/data/y2_data.npy', y2_data)