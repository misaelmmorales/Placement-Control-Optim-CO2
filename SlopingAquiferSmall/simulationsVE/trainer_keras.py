import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras import Model, layers, regularizers, optimizers, losses, callbacks, metrics

from pix2vid2 import make_model, MonitorCallback, CustomLoss

msteps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

NUM_REALIZATIONS = 1000
NX,  NY,  NZ = 64, 64, 1
NTT, NT1, NT2 = 40, 20, len(msteps)
X_CHANNELS  = 5
Y1_CHANNELS = 2
Y2_CHANNELS = 1
HIDDEN = [16, 64, 256]

NTRAIN = 600
EPOCHS = 100
MONITOR = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-8
PATIENCE = 20

sec2year   = 365.25 * 24 * 60 * 60
Darcy      = 9.869233e-13
psi2pascal = 6894.76
co2_rho    = 686.5266
milli      = 1e-3
mega       = 1e6

def check_tf_gpu():
    sys_info = tf.sysconfig.get_build_info()
    kversion = keras.__version__
    version, cuda, cudnn = tf.__version__, sys_info["cuda_version"], sys_info["cudnn_version"]
    count = len(tf.config.experimental.list_physical_devices())
    name  = [device.name for device in tf.config.experimental.list_physical_devices('GPU')]
    print('-'*62)
    print('------------------------ VERSION INFO ------------------------')
    print('TF version: {} | Keras: {} | # Device(s) available: {}'.format(version, kversion, count))
    print('TF Built with CUDA? {} | CUDA: {} | cuDNN: {}'.format(tf.test.is_built_with_cuda(), cuda, cudnn))
    print(tf.config.list_physical_devices()[-1])
    print('-'*62+'\n')
    return None
check_tf_gpu()

deltatime = sio.loadmat('data/time_arr.mat', simplify_cells=True)['time_arr']
timesteps = np.cumsum(deltatime)
timesteps_inj = timesteps[:20]
timesteps_mon = timesteps[20:][msteps]
print('timesteps: {} | deltatime: {}'.format(len(timesteps), np.unique(deltatime)))
print('injection: {}'.format(timesteps_inj))
print('monitoring: {}'.format(timesteps_mon))

# Load data
X_data = np.load('data/X_data.npy')   # (poro,perm,well,tops,heights)
c_data = np.load('data/c_data.npy')   # (controls)
y1_data = np.load('data/y1_data.npy') # (pressure,saturation)_inj
y2_data = np.load('data/y2_data.npy') # (saturation)_monitor
print('X: {} | c: {}'.format(X_data.shape, c_data.shape))
y2_data = y2_data[:,msteps]
#y2_data = y2_data[:, [0, 1, 2, 3, 4, 9, 14, 19]]
print('y1: {} | y2: {}'.format(y1_data.shape, y2_data.shape))

# Normalize data
pmu, psd = X_data[...,0].mean(), X_data[...,0].std() # porosity
kmu, ksd = X_data[...,1].mean(), X_data[...,1].std() # permeability
wmi, wma = X_data[...,2].min(),  X_data[...,2].max() # wells
tmi, tma = X_data[...,3].min(),  X_data[...,3].max() # tops
vmi, vma = X_data[...,4].min(),  X_data[...,4].max() # heights
cmi, cma = c_data.min(),         c_data.max()        # controls

X_data[...,0] = (X_data[...,0] - pmu) / (3.33*psd)
X_data[...,1] = (X_data[...,1] - kmu) / (3.33*ksd)
X_data[...,2] = (X_data[...,2] - wmi) / (wma - wmi)
X_data[...,3] = (X_data[...,3] - tmi) / (tma - tmi)
X_data[...,4] = (X_data[...,4] - vmi) / (vma - vmi)
c_data = c_data / 5.0

y1_data[...,0]  = y1_data[...,0]  / 5e5
y1_data[...,-1] = y1_data[...,-1] / 1
y2_data[...,-1] = y2_data[...,-1] / 1

print('porosity     - min: {:.2f} | max: {:.2f}'.format(X_data[...,0].min(), X_data[...,0].max()))
print('logperm      - min: {:.2f} | max: {:.2f}'.format(X_data[...,1].min(), X_data[...,1].max()))
print('wells        - min: {:.2f} | max: {:.2f}'.format(X_data[...,2].min(), X_data[...,2].max()))
print('tops         - min: {:.2f} | max: {:.2f}'.format(X_data[...,3].min(), X_data[...,3].max()))
print('volumes      - min: {:.2f} | max: {:.2f}'.format(X_data[...,4].min(), X_data[...,4].max()))
print('controls     - min: {:.2f} | max: {:.2f}'.format(c_data.min(),        c_data.max()))
print('pressure_1   - min: {:.2f} | max: {:.2f}'.format(y1_data[...,0].min(), y1_data[...,0].max()))
print('saturation_1 - min: {:.2f} | max: {:.2f}'.format(y1_data[...,-1].min(), y2_data[...,-1].max()))
print('saturation_2 - min: {:.2f} | max: {:.2f}'.format(y2_data[...,-1].min(), y2_data[...,-1].max()))

train_idx = np.random.choice(range(len(X_data)), NTRAIN, replace=False)
np.save('models/training_idx.npy', train_idx)
print('Training index saved!')
#np.load('models/training_idx.npy')
test_idx  = np.setdiff1d(range(len(X_data)), train_idx)

# X_train = X_data[train_idx].astype(np.float32)
# c_train = c_data[train_idx].astype(np.float32)
# y1_train = y1_data[train_idx].astype(np.float32)
# y2_train = y2_data[train_idx].astype(np.float32)
# X_test = X_data[test_idx].astype(np.float32)
# c_test = c_data[test_idx].astype(np.float32)
# y1_test = y1_data[test_idx].astype(np.float32)
# y2_test = y2_data[test_idx].astype(np.float32)

X_train  = tf.cast(X_data[train_idx],  tf.float32)
c_train  = tf.cast(c_data[train_idx],  tf.float32)
y1_train = tf.cast(y1_data[train_idx], tf.float32)
y2_train = tf.cast(y2_data[train_idx], tf.float32)
X_test   = tf.cast(X_data[test_idx],   tf.float32)
c_test   = tf.cast(c_data[test_idx],   tf.float32)
y1_test  = tf.cast(y1_data[test_idx],  tf.float32)
y2_test  = tf.cast(y2_data[test_idx],  tf.float32)

print('-'*70)
print('X_train:  {}     | c_train: {}'.format(X_train.shape, c_train.shape))
print('y1_train: {} | y2_train: {}'.format(y1_train.shape, y2_train.shape))
print('-'*70)
print('X_test:  {}     | c_test: {}'.format(X_test.shape, c_test.shape))
print('y1_test: {} | y2_test: {}'.format(y1_test.shape, y2_test.shape))
print('-'*70)

esCallback = callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)
mcCallback = callbacks.ModelCheckpoint('pix2vid-opt-v2.keras', monitor='val_accuracy', save_best_only=True)
customCBs  = [MonitorCallback(monitor=10), esCallback, mcCallback]

model1, model2 = make_model(hidden=HIDDEN)

optimizer1 = optimizers.Adam(learning_rate=LEARNING_RATE)#, weight_decay=WEIGHT_DECAY)
model1.compile(optimizer=optimizer1, loss=CustomLoss(b=0.75, a=0.4), metrics=['mse'])

optimizer2 = optimizers.Adam(learning_rate=LEARNING_RATE)#, weight_decay=WEIGHT_DECAY)
model2.compile(optimizer=optimizer2, loss=CustomLoss(b=0.75, a=0.4), metrics=['mse'])

start = time()
fit1 = model1.fit(x=[X_train, c_train], y=y1_train,
                batch_size       = BATCH_SIZE,
                epochs           = EPOCHS,
                validation_split = 0.25,
                shuffle          = True,
                callbacks        = [MonitorCallback(monitor=MONITOR)],
                verbose          = 0)
print('-'*30+'\n'+'Training time: {:.3f} minutes'.format((time()-start)/60))
model1.save('models/pix2vid-v2-inj.keras')
model1.save_weights('models/pix2vid-v2-inj.weights.h5')
pd.DataFrame(fit1.history).to_csv('models/pix2vid-v2-inj.csv', index=False)

start = time()
fit2 = model2.fit(x=[X_train, c_train], y=y2_train,
                batch_size       = BATCH_SIZE,
                epochs           = EPOCHS,
                validation_split = 0.25,
                shuffle          = True,
                callbacks        = [MonitorCallback(monitor=MONITOR)],
                verbose          = 0)
model2.save('models/pix2vid-v2-mon.keras')
model2.save_weights('models/pix2vid-v2-mon.weights.h5')
pd.DataFrame(fit2.history).to_csv('models/pix2vid-v2-mon.csv', index=False)

print('... Done!')
######################################## END ########################################