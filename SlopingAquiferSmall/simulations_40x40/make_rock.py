import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

import scipy.io as sio

fnames = []
for root, dirs, files in os.walk('/mnt/e/MLTrainingImages'):
    for file in files:
        if file.endswith('.npy'):
            fnames.append(os.path.join(root, file))

all_facies = np.zeros((len(fnames), 256, 256, 128))
for i in tqdm(range(len(fnames))):
    f = np.load(fnames[i]).reshape(256,256,128)
    all_facies[i] = f

all_f = []
for i in tqdm(range(len(fnames))):
    f1 = resize(all_facies[i, :, :, 24:29], (40, 40, 5))
    f2 = resize(all_facies[i, :, :, 48:53], (40, 40, 5))
    f3 = resize(all_facies[i, :, :, 72:79], (40, 40, 5))
    f4 = resize(all_facies[i, :, :, 96:101], (40, 40, 5))
    all_f.append(np.stack([f1, f2, f3, f4], axis=-1))

all_f = np.array(all_f)
all_f = np.moveaxis(all_f, -1, 1).reshape(-1, 40, 40, 5)
print(all_f.shape)
print('facies - min: {:.4f} | max: {:.4f}'.format(all_f.min(), all_f.max()))

scaler = MinMaxScaler((0.75, 1.25))
facies_norm = scaler.fit_transform(all_f.reshape(1272, -1)).reshape(-1, 40, 40, 5)
print('facies_norm - shape: {} | min: {:.4f} | max: {:.4f}'.format(facies_norm.shape, facies_norm.min(), facies_norm.max()))

all_perm = np.moveaxis(np.array(pd.read_csv('perm_40x40x5.csv')).reshape(40,40,5,1272,order='F'), -1, 0)
print(all_perm.shape)
print('perm - min: {:.4f} | max: {:.4f}'.format(all_perm.min(), all_perm.max()))

all_perm = MinMaxScaler((np.log10(0.1), np.log10(2566))).fit_transform(all_perm.reshape(1272, -1)).reshape(-1, 40, 40, 5)
print('perm - min:  {:.4f} | max: {:.4f}'.format(all_perm.min(), all_perm.max()))

all_poro = 10**((all_perm-8.66)/10)
print('poro - min:  {:.4f} | max: {:.4f}'.format(all_poro.min(), all_poro.max()))

poro_scaled = all_poro * facies_norm
print('poro_scaled - min: {:.4f} | max: {:.4f}'.format(poro_scaled.min(), poro_scaled.max()))

perm_scaled = np.log10(poro_scaled) * 6 + 6
print('perm_scaled - min: {:.4f} | max: {:.4f}'.format(perm_scaled.min(), perm_scaled.max()))
print('perm_scaled - min: {:.4f} | max: {:.4f}'.format(10**perm_scaled.min(), 10**perm_scaled.max()))

fig, axs = plt.subplots(3, 10, figsize=(20,4), sharex=True, sharey=True)
for j in range(10):
    ax1, ax2, ax3 = axs[0, j], axs[1, j], axs[2,j]
    k = j * 125
    im1 = ax1.imshow(poro_scaled[k, :, :, 0], cmap='turbo')
    ax1.set_title('R{}'.format(k))
    im2 = ax2.imshow(perm_scaled[k, :, :, 0], cmap='turbo')
    im3 = ax3.imshow(facies_norm[k, :, :, 0], cmap='turbo')
    [plt.colorbar(ii, pad=0.04, fraction=0.046) for ii in [im1, im2, im3]]
    [a.axis('off') for a in [ax1, ax2, ax3]]
plt.tight_layout()
plt.savefig('facies_perm_poro_40x40.png', dpi=600)
plt.close()

for i in tqdm(range(1272)):
    sio.savemat('simulations_40x40/rock/mat/rock_{}.mat'.format(i), {'poro': poro_scaled[i], 'perm': perm_scaled[i], 'facies': facies_norm[i]})

############################## END ########################################