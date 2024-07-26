import numpy as np
from scipy.io import loadmat
from multiprocessing import Pool

n_timesteps = 33
nx, ny, nz = 100, 100, 11
indexMap = loadmat('data_100_100_11/G_cells_indexMap.mat', simplify_cells=True)['gci']
Grid = np.zeros((nx,ny,nz)).flatten(order='F')
Grid[indexMap] = 1
Grid = Grid.reshape(nx,ny,nz, order='F')

### inputs ###
for i in range(1272):
    r = loadmat('data_100_100_11/rock/rock_{}.mat'.format(i))
    poro, perm = r['poro'], r['perm']

    wid = np.zeros((2,5), dtype='int')
    warr = loadmat('data_100_100_11/well_coords/well_coords_{}.mat'.format(i), simplify_cells=True)['var']
    wlist = np.array(list(warr.values()))
    wlist = wlist.reshape(-1, 1) if wlist.ndim == 1 else wlist
    for j in range(wlist.shape[-1]):
        wid[:,j] = wlist[:,j]

    d = loadmat('data_100_100_11/states/states_{}.mat'.format(i), simplify_cells=True)['var']

    timesteps = np.zeros((n_timesteps,1))
    for tt in range(n_timesteps):
        timesteps[tt] = d[tt+1]['time']

    wctrl = np.zeros((n_timesteps,5))
    for t in range(1,n_timesteps+1):
        for w in range(wlist.shape[-1]):
            if wlist.shape[-1] == 1:
                wctrl[t-1, w] = d[t]['wellSol']['val']
            else:
                wctrl[t-1, w] = d[t]['wellSol'][w]['val']

    np.savez('data_npy_100_100_11/inputs_rock_rates_locs_time/x_{}.npz'.format(i), 
             poro=poro, perm=perm, locs=wid, wlist=wlist, ctrl=wctrl, time=timesteps)

### Linear outputs ###
def process_file(i):
    d = loadmat('data_100_100_11/states/states_{}.mat'.format(i), simplify_cells=True)['var']
    p = np.zeros((n_timesteps+1, 29128))
    s = np.zeros((n_timesteps+1, 29128))
    for t in range(n_timesteps+1):
        p[t] = d[t]['pressure']
        s[t] = d[t]['s'][:,1]
    np.savez('data_npy_100_100_11/outputs_pressure_saturation/y_{}.npz'.format(i), pressure=p, saturation=s)

file_indeces = list(range(1272))
pool = Pool(processes=8)
pool.map(process_file, file_indeces)
pool.close()
pool.join()

### 3D outputs ###
def process_file(i):
    dd = np.load('data_npy_100_100_11/outputs_masked_pressure_saturation/y_{}.npz'.format(i))

    G_expanded = np.repeat(np.expand_dims(Grid, 0), n_timesteps, 0)

    pp = np.zeros((n_timesteps, nx, ny, nz))
    ss = np.zeros((n_timesteps, nx, ny, nz))
    for j in range(n_timesteps):
        p = np.zeros((nx,ny,nz)).flatten(order='F')
        s = np.zeros((nx,ny,nz)).flatten(order='F')
        p[indexMap] = dd['pressure'][j+1]
        s[indexMap] = dd['saturation'][j+1]
        pp[j] = p.reshape(nx,ny,nz, order='F')
        ss[j] = s.reshape(nx,ny,nz, order='F')
        
    pressure = np.ma.masked_where(G_expanded==0, pp)
    saturation = np.ma.masked_where(G_expanded==0, ss)

    np.savez('data_npy_100_100_11/outputs_pressure_saturation/y_{}.npz'.format(i), 
             pressure=pressure, saturation=saturation)
    
file_indices = list(range(1272))
pool = Pool(8)
pool.map(process_file, file_indices)
pool.close()
pool.join()

### NPZ outputs ###
def process_npz_file(i)
    w = np.zeros((100,100,11))
    m = np.load('data_npy_100_100_11/inputs_rock_rates_locs_time/x_{}.npz'.format(i))

    p = np.expand_dims(apply_mask(m['poro']), 0)[...,5:10] / (0.3)
    k = np.expand_dims(apply_mask(m['perm']), 0)[...,5:10] / (3.3)

    w[m['wlist'][0,:]-1, m['wlist'][1,:]-1, :] = 1
    w = np.expand_dims(apply_mask(w), 0)[...,5:10]

    t = np.expand_dims(apply_mask(Tops), 0)[...,5:10]      / (Tops.max())
    g = np.expand_dims(Grid, 0)[...,5:10]

    xm = np.concatenate([p,k,w,t,g], axis=0)
    xc = m['ctrl'] * co2_rho*sec2year/mega/1e3             / (25)
    xt = m['time']                                         / sec2year/(110)

    dd = np.load('data_npy_100_100_11/outputs_pressure_saturation/y_{}.npz'.format(i))

    prm = dd['pressure'][...,5:10]                         / (psi2pascal) / (40000)
    sam = dd['saturation'][...,5:10]
    yy  = np.stack([prm,sam], axis=1)

    prf = dd['pressure'].reshape(33, -1, order='F')[:,indexMap] /psi2pascal/40000
    srf = dd['saturation'].reshape(33, -1, order='F')[:,indexMap]
    yf  = np.stack([prf,srf], axis=1)

    np.savez('data/realization_{}.npz'.format(i), xm=xm, xc=xc, xt=xt, yy=yy)