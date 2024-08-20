import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

import keras
import tensorflow as tf
from pix2vid2 import make_model

sample = 1

NUM_REALIZATIONS = 929
NX,  NY,  NZ = 64, 64, 1
NTT, NT1, NT2 = 40, 20, 5
HIDDEN = [16, 64, 256]

sec2year   = 365.25 * 24 * 60 * 60
Darcy      = 9.869233e-13
psi2pascal = 6894.76
co2_rho    = 686.5266
milli      = 1e-3
mega       = 1e6

deltatime = sio.loadmat('simulations/data/time_arr.mat', simplify_cells=True)['time_arr']
timesteps = np.cumsum(deltatime)
timesteps_inj = timesteps[:20]
timesteps_mon = timesteps[[21, 24, 29, 34, 39]]

print('timesteps: {} | deltatime: {}'.format(len(timesteps), np.unique(deltatime)))
print('injection: {}'.format(timesteps_inj))
print('monitoring: {}'.format(timesteps_mon))

# Load data
X_data = np.load('simulations/data/X_data.npy')
c_data = np.load('simulations/data/c_data.npy')
y1_data = np.load('simulations/data/y1_data.npy')
y2_data = np.load('simulations/data/y2_data.npy')[:,[1, 4, 9, 14, 19]]
print('X: {} | c: {}'.format(X_data.shape, c_data.shape))
print('y1: {} | y2: {}'.format(y1_data.shape, y2_data.shape))

# Normalize data
pmu, psd = X_data[...,0].mean(), X_data[...,0].std() # porosity
kmu, ksd = X_data[...,1].mean(), X_data[...,1].std() # permeability
wmi, wma = X_data[...,2].min(),  X_data[...,2].max() # wells
tmi, tma = X_data[...,3].min(),  X_data[...,3].max() # tops
vmi, vma = X_data[...,4].min(),  X_data[...,4].max() # volumes
cmi, cma = c_data.min(),         c_data.max()        # controls

X_data[...,0] = (X_data[...,0] - pmu) / (3.33*psd)
X_data[...,1] = (X_data[...,1] - kmu) / (3.33*ksd)
X_data[...,2] = (X_data[...,2] - wmi) / (wma - wmi)
X_data[...,3] = (X_data[...,3] - tmi) / (tma - tmi)
X_data[...,4] = (X_data[...,4] - vmi) / (vma - vmi)
c_data = c_data / 2.0

y1_data[...,0]  = y1_data[...,0]  / 50e3
y1_data[...,-1] = y1_data[...,-1] / 0.73
y2_data[...,-1] = y2_data[...,-1] / 0.73

print('porosity     - min: {:.2f} | max: {:.2f}'.format(X_data[...,0].min(), X_data[...,0].max()))
print('logperm      - min: {:.2f} | max: {:.2f}'.format(X_data[...,1].min(), X_data[...,1].max()))
print('wells        - min: {:.2f} | max: {:.2f}'.format(X_data[...,2].min(), X_data[...,2].max()))
print('tops         - min: {:.2f} | max: {:.2f}'.format(X_data[...,3].min(), X_data[...,3].max()))
print('volumes      - min: {:.2f} | max: {:.2f}'.format(X_data[...,4].min(), X_data[...,4].max()))
print('controls     - min: {:.2f} | max: {:.2f}'.format(c_data.min(),        c_data.max()))
print('pressure_1   - min: {:.2f} | max: {:.2f}'.format(y1_data[...,0].min(), y1_data[...,0].max()))
print('saturation_1 - min: {:.2f} | max: {:.2f}'.format(y1_data[...,-1].min(), y2_data[...,-1].max()))
print('saturation_2 - min: {:.2f} | max: {:.2f}'.format(y2_data[...,-1].min(), y2_data[...,-1].max()))

train_idx = np.load('models/training_idx.npy')
test_idx  = np.setdiff1d(range(len(X_data)), train_idx)

X_train = X_data[train_idx].astype(np.float32)
c_train = c_data[train_idx].astype(np.float32)
y1_train = y1_data[train_idx].astype(np.float32)
y2_train = y2_data[train_idx].astype(np.float32)
X_test = X_data[test_idx].astype(np.float32)
c_test = c_data[test_idx].astype(np.float32)
y1_test = y1_data[test_idx].astype(np.float32)
y2_test = y2_data[test_idx].astype(np.float32)

print('X_train:  {}     | c_train: {}'.format(X_train.shape, c_train.shape))
print('y1_train: {} | y2_train: {}'.format(y1_train.shape, y2_train.shape))
print('-'*70)
print('X_test:  {}     | c_test: {}'.format(X_test.shape, c_test.shape))
print('y1_test: {} | y2_test: {}'.format(y1_test.shape, y2_test.shape))

losses = pd.read_csv('models/pix2vid-v2.csv')
plt.figure(figsize=(8,4))
plt.plot(losses.index, losses['loss'], label='Train')
plt.plot(losses.index, losses['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both')
plt.savefig('figures/pix2vid-v2_training-performance.png', dpi=600)
plt.close()

model = make_model(hidden=HIDDEN)
model.load_weights('models/pix2vid-v2.weights.h5')

xtrain = tf.cast(X_train[:20], tf.float32)
ctrain = tf.cast(c_train[:20], tf.float32)
y1train = tf.cast(y1_train[:20], tf.float32)
y2train = tf.cast(y2_train[:20], tf.float32)

xtest = tf.cast(X_test[:20], tf.float32)
ctest = tf.cast(c_test[:20], tf.float32)
y1test = tf.cast(y1_test[:20], tf.float32)
y2test = tf.cast(y2_test[:20], tf.float32)

y1train_pred, y2train_pred = model.predict([xtrain, ctrain], verbose=False)
y1train_pred, y2train_pred = np.array(y1train_pred), np.array(y2train_pred)

y1test_pred, y2test_pred = model.predict([xtest, ctest], verbose=False)
y1test_pred, y2test_pred = np.array(y1test_pred), np.array(y2test_pred)

fig, axs = plt.subplots(5, 10, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax = axs[i,j]
        im = ax.imshow(xtrain[j,...,i], cmap='turbo')
        plt.colorbar(im, pad=0.04, fraction=0.046)
        ax.set_title('R{}'.format(j)) if i == 0 else None
plt.tight_layout()
plt.savefig('figures/pix2vid-v2_training-inputs.png', dpi=600)
plt.close()

fig, axs = plt.subplots(1, 10, figsize=(15,6), sharex=True, sharey=True)
for j in range(10):
    ax = axs[j]
    im = ax.imshow(ctrain[j], cmap='turbo')
    ax.set_xticks(range(5), labels=np.arange(1,6))
    ax.set_yticks(range(20), labels=np.arange(1,21))
    ax.set(title='R{}'.format(j))
plt.tight_layout()
plt.savefig('figures/pix2vid-v2_training-controls.png', dpi=600)
plt.close()

plt.figure(figsize=(12,4))
for j in range(10):
    plt.subplot(1, 10, j+1)
    k = j*2+1
    plt.imshow(y1train[sample, k, ..., -1], 'jet')
    plt.title('t={}'.format(timesteps_inj[k]))
plt.tight_layout()
plt.savefig('figures/pix2vid-v2_training-injection-true.png', dpi=600)
plt.close()

plt.figure(figsize=(12,4))
for j in range(10):
    plt.subplot(1, 10, j+1)
    k = j*2+1
    plt.imshow(y1train_pred[sample, k, ..., -1], 'jet')
    plt.title('t={}'.format(timesteps_inj[k]))
plt.tight_layout()
plt.savefig('figures/pix2vid-v2_training-injection-predicted.png', dpi=600)
plt.close()

plt.figure(figsize=(12,4))
for j in range(5):
    plt.subplot(1, 5, j+1)
    plt.imshow(y2train[sample, j, ..., -1], 'jet')
    plt.title('t={}'.format(timesteps_mon[j]))
plt.tight_layout()
plt.savefig('figures/pix2vid-v2_training-monitoring-true.png', dpi=600)
plt.close()

plt.figure(figsize=(12,4))
for j in range(5):
    plt.subplot(1, 5, j+1)
    plt.imshow(y2train_pred[sample, j, ..., -1], 'jet')
    plt.title('t={}'.format(timesteps_mon[j]))
plt.tight_layout()
plt.savefig('figures/pix2vid-v2_training-monitoring-predicted.png', dpi=600)
plt.close()