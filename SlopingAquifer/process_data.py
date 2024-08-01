import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from multiprocessing import Pool
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

NT, NT0    = 60, 30
NX, NY, NZ = 160, 160, 5
DX, DY, DZ = 250, 250, 40
sec2year   = 365.25 * 24 * 60 * 60
psi2pascal = 6894.76
co2_rho    = 686.5266
mega       = 1e6
mt2m3      = sec2year*co2_rho/mega/1e3

n_realiations = 318
n_slices = 4
slice1, slice2, slice3, slice4 = 26, 55, 75, 102

gridVE = sio.loadmat('Gt.mat', simplify_cells=True)['Gt']
tops2d = -gridVE['cells']['z'].reshape(NX,NY)

fnames = []
for root, dirs, files in os.walk('/mnt/e/MLTrainingImages'):
    for file in files:
        if file.endswith('.npy'):
            fnames.append(os.path.join(root,file))
print(len(fnames))

def process_facies_realization(idx, range=(0.25,1.2)):
    dd = np.load(fnames[idx]).reshape(256,256,128)
    f0 = np.rot90(resize(dd[..., slice1:slice1+NZ], (NX,NY), anti_aliasing=True, preserve_range=True), 1, (0,1))
    f1 = np.rot90(resize(dd[..., slice2:slice2+NZ], (NX,NY), anti_aliasing=True, preserve_range=True), 1, (0,1))
    f2 = np.rot90(resize(dd[..., slice3:slice3+NZ], (NX,NY), anti_aliasing=True, preserve_range=True), 1, (0,1))
    f3 = np.rot90(resize(dd[..., slice4:slice4+NZ], (NX,NY), anti_aliasing=True, preserve_range=True), 1, (0,1))
    facies_all = np.stack([f0, f1, f2, f3], axis=0)
    scaler = MinMaxScaler(range)
    facies_all = scaler.fit_transform(facies_all.reshape(n_slices, -1)).reshape(n_slices, NX, NY, NZ)
    return facies_all

facies = np.zeros((n_realiations, n_slices, NX, NY, NZ))
for i in tqdm(range(n_realiations), desc='Processing facies'):
    dd = process_facies_realization(i)
    facies[i] = dd

facies = facies.reshape(n_realiations*n_slices, NX, NY, NZ)

for i in tqdm(range(n_realiations*n_slices), desc='Saving facies'):
    sio.savemat('/mnt/e/Placement-Control-Optim-CO2/IGEM/facies/mat/facies_{}.mat'.format(i), {'facies': facies[i]})
    np.save('/mnt/e/Placement-Control-Optim-CO2/IGEM/facies/npy/facies_{}.npy'.format(i), facies[i])

k1 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b1')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k2 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b2')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k3 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b3')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k4 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b4')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k5 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b5')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k6 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b6')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k7 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b7')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k8 = np.moveaxis(np.array(pd.read_csv('rock/perm_160x160x5_n159_b8')).reshape(NX,NY,NZ,159,order='F'), -1, 0)
k_all = np.concatenate([k1, k2, k3, k4, k5, k6, k7, k8], axis=0)
for i in tqdm(range(n_realiations*n_slices), desc='Processing Rock'):
    f = np.load('/mnt/e/Placement-Control-Optim-CO2/IGEM/facies/npy/facies_{}.npy'.format(i))
    k = k_all[i]
    kk = np.log10(10**(k+3.3) * f) /  1.6
    pp = 10**((kk-9)/10)
    rock = np.stack([kk.flatten(order='F'), pp.flatten(order='F')], axis=0)
    np.savez('/mnt/e/Placement-Control-Optim-CO2/IGEM/rock/npz/rock_{}.npz'.format(i), perm=kk, poro=pp, facies=f, rock=rock)
    sio.savemat('/mnt/e/Placement-Control-Optim-CO2/IGEM/rock/mat/rock_{}.mat'.format(i), {'perm':kk, 'poro':pp, 'facies':f})

def process_rock(i):
    f = np.load('facies/npy/facies_{}.npy'.format(i))
    f = np.clip(f, 0.25, 1.2)
    k = k_all[i]
    k = np.clip(k, -10, None)
    kk = np.log10(10**(k+3.3) * f) /  1.6
    pp = 10**((kk-9)/10)
    if any(np.isnan(kk.flatten())):
        print('k NAN:', i)
    if any(np.isnan(pp.flatten())):
        print('p NAN:', i)
    rock = np.stack([kk.flatten(order='F'), pp.flatten(order='F')], axis=0)
    np.savez('rock/npz/rock_{}.npz'.format(i), perm=kk, poro=pp, facies=f, rock=rock)
    sio.savemat('rock/mat/rock_{}.mat'.format(i), {'perm':kk, 'poro':pp, 'facies':f})

def process_states(i):
    dd = sio.loadmat('states/states_{}.mat'.format(i), simplify_cells=True)['var']
    pressure = np.zeros((NT,NX,NY))
    saturation = np.zeros((NT,NX,NY))
    for j in range(NT):
        pressure[j] = dd[j]['pressure'].reshape(NX,NY) / psi2pascal
        saturation[j] = dd[j]['s'].reshape(NX,NY)
    np.save('states/pressure/pressure_{}.npy'.format(i), pressure)
    np.save('states/saturation/saturation_{}.npy'.format(i), saturation)

def process_data(i):
    tops = np.repeat(np.expand_dims(tops2d, -1), NT0, axis=-1)
    rock = np.load('rock/VE2d/npz/rock2d_{}.npz'.format(i))
    perm = np.repeat(np.expand_dims(rock['perm'].reshape(NX,NY), -1), NT0, axis=-1)
    poro = np.repeat(np.expand_dims(rock['poro'].reshape(NX,NY), -1), NT0, axis=-1)
    rock.close()

    pressure   = np.moveaxis(np.load('states/pressure/pressure_{}.npy'.format(i)), 0, -1)
    saturation = np.moveaxis(np.load('states/saturation/saturation_{}.npy'.format(i)), 0, -1)

    ww = sio.loadmat('well_locs/well_locs_{}.mat'.format(i), simplify_cells=True)['var'] - 1
    cc = sio.loadmat('controls/controls_{}.mat'.format(i), simplify_cells=True)['var'] * sec2year * co2_rho / 1e3 / mega

    if len(cc.shape) == 1:
        cc = cc.reshape(1, -1)
    if len(ww.shape) == 1:
        ww = ww.reshape(1, -1)

    well_locs = np.zeros((NX,NY))
    well_locs[ww[:,1], ww[:,0]] = 1
    well_locs = np.repeat(np.expand_dims(well_locs, -1), NT0, axis=-1)

    controls = np.zeros((NX,NY,NT0))
    controls[ww[:,1], ww[:,0], :] = cc

    features = np.stack([poro, perm, tops, well_locs, controls], axis=0)
    labels = np.stack([pressure, saturation], axis=0)

    np.savez('data/realiation_{}.npz'.format(i), features=features, labels1=labels[...,:NT0], labels2=labels[...,NT0:])

def run_parallel_processing(iterations, num_processes):
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_rock, iterations), total=len(iterations), desc='Processing Rock-to-.mat'))

iterations = list(range(n_realiations*n_slices))
run_parallel_processing(iterations, num_processes=8)

print('-'*50)
print(' '*23, 'Done', ' '*23)
print('-'*50)