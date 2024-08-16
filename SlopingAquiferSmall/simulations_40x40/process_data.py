import numpy as np
import scipy.io as sio
from tqdm import tqdm

NR, NT = 1272, 40
NX, NY = 40, 40
milli  = 1e-3
Darcy  = 9.869233e-13
psi2pa = 6894.75729
tops2d = sio.loadmat('simulations_40x40/Gt.mat', simplify_cells=True)['Gt']['cells']['z'].reshape(NX,NY,order='F')

poro = np.zeros((NR, NX, NY))
perm = np.zeros((NR, NX, NY))
tops = np.repeat(np.expand_dims(tops2d, axis=0), NR, axis=0)
well = np.zeros((NR, NX, NY))
ctrl = np.zeros((NR, 5, NT//2))
for i in tqdm(range(NR)):
    r = sio.loadmat('simulations_40x40/rock/VE2d/rock2d_{}.mat'.format(i), simplify_cells=True)['var']
    w = sio.loadmat('simulations_40x40/well_locs/well_locs_{}.mat'.format(i), simplify_cells=True)['var']
    c = sio.loadmat('simulations_40x40/controls/controls_{}.mat'.format(i), simplify_cells=True)['var']

    if len(w.shape)==1:
        w = w.reshape(1, 2)
    if len(c.shape)==1:
        c = c.reshape(1, -1)
    nw = w.shape[0]

    k = np.log10(r['perm']/milli/Darcy)
    poro[i] = r['poro'].reshape(NX, NY, order='F')
    perm[i] = k.reshape(NX, NY, order='F')
    well[i, w[:,0], w[:,1]] = 1
    ctrl[i, :nw] = c

print('poro: {} | perm: {} | tops: {} | well: {}'.format(poro.shape, perm.shape, tops.shape, well.shape))
print('ctrl: {}'.format(ctrl.shape))

X_data = np.stack([poro,perm,tops,well], axis=1)
c_data = np.moveaxis(ctrl, -1, 1)
print('X_data: {} | c_data: {}'.format(X_data.shape, c_data.shape))
np.save('simulations_40x40/X_data.npy', X_data)
np.save('simulations_40x40/c_data.npy', c_data)

pressure = np.zeros((NR, NT, NX, NY))
saturation = np.zeros((NR, NT, NX, NY))
for i in tqdm(range(NR)):
    d = sio.loadmat('simulations_40x40/states/states_{}.mat'.format(i), simplify_cells=True)['var']
    for j in range(NT):
        pressure[i,j] = d[j]['pressure'].reshape(NX, NY, order='F') / psi2pa
        saturation[i,j] = d[j]['s'].reshape(NX, NY, order='F')
y_data = np.stack([pressure, saturation], axis=2)

y1_data = y_data[:,:20]
y2_data = np.expand_dims(y_data[:,20:,-1], axis=2)
np.save('simulations_40x40/y1_data.npy', y1_data)
np.save('simulations_40x40/y2_data.npy', y2_data)