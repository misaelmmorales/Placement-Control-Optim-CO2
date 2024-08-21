import numpy as np
import keras
import tensorflow as tf

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

def describe_data(X_data, c_data, y1_data, y2_data):
    print('X: {} | c: {}'.format(X_data.shape, c_data.shape))
    print('y1: {} | y2: {}'.format(y1_data.shape, y2_data.shape))
    print('porosity     - min: {:.2f} | max: {:.2f}'.format(X_data[...,0].min(), X_data[...,0].max()))
    print('logperm      - min: {:.2f} | max: {:.2f}'.format(X_data[...,1].min(), X_data[...,1].max()))
    print('wells        - min: {:.2f} | max: {:.2f}'.format(X_data[...,2].min(), X_data[...,2].max()))
    print('tops         - min: {:.2f} | max: {:.2f}'.format(X_data[...,3].min(), X_data[...,3].max()))
    print('volumes      - min: {:.2f} | max: {:.2f}'.format(X_data[...,4].min(), X_data[...,4].max()))
    print('controls     - min: {:.2f} | max: {:.2f}'.format(c_data.min(),        c_data.max()))
    print('pressure_1   - min: {:.2f} | max: {:.2f}'.format(y1_data[...,0].min(), y1_data[...,0].max()))
    print('saturation_1 - min: {:.2f} | max: {:.2f}'.format(y1_data[...,-1].min(), y2_data[...,-1].max()))
    print('saturation_2 - min: {:.2f} | max: {:.2f}'.format(y2_data[...,-1].min(), y2_data[...,-1].max()))
    return None

def load_data(folder=None, cmax=5.0, presmax=5e5, satmax=1.0, pfactor=3.33, kfactor=3.33):
    X_data = np.load('{}/data/X_data.npy'.format(folder))   # (poro,perm,well,tops,heights)
    c_data = np.load('{}/data/c_data.npy'.format(folder))   # (controls)
    y1_data = np.load('{}/data/y1_data.npy'.format(folder)) # (pressure,saturation)_inj
    y2_data = np.load('{}/data/y2_data.npy'.format(folder)) # (saturation)_monitor
    print('X: {} | c: {}'.format(X_data.shape, c_data.shape))
    print('y1: {} | y2: {}'.format(y1_data.shape, y2_data.shape))

    pmu, psd = X_data[...,0].mean(), X_data[...,0].std() # porosity
    kmu, ksd = X_data[...,1].mean(), X_data[...,1].std() # permeability
    wmi, wma = X_data[...,2].min(),  X_data[...,2].max() # wells
    tmi, tma = X_data[...,3].min(),  X_data[...,3].max() # tops
    vmi, vma = X_data[...,4].min(),  X_data[...,4].max() # heights
    cmi, cma = c_data.min(),         c_data.max()        # controls

    X_data[...,0] = (X_data[...,0] - pmu) / (pfactor*psd)
    X_data[...,1] = (X_data[...,1] - kmu) / (kfactor*ksd)
    X_data[...,2] = (X_data[...,2] - wmi) / (wma - wmi)
    X_data[...,3] = (X_data[...,3] - tmi) / (tma - tmi)
    X_data[...,4] = (X_data[...,4] - vmi) / (vma - vmi)
    c_data = c_data / cmax

    y1_data[...,0]  = y1_data[...,0]  / presmax
    y1_data[...,-1] = y1_data[...,-1] / satmax
    y2_data[...,-1] = y2_data[...,-1] / satmax

    print('porosity     - min: {:.2f} | max: {:.2f}'.format(X_data[...,0].min(), X_data[...,0].max()))
    print('logperm      - min: {:.2f} | max: {:.2f}'.format(X_data[...,1].min(), X_data[...,1].max()))
    print('wells        - min: {:.2f} | max: {:.2f}'.format(X_data[...,2].min(), X_data[...,2].max()))
    print('tops         - min: {:.2f} | max: {:.2f}'.format(X_data[...,3].min(), X_data[...,3].max()))
    print('volumes      - min: {:.2f} | max: {:.2f}'.format(X_data[...,4].min(), X_data[...,4].max()))
    print('controls     - min: {:.2f} | max: {:.2f}'.format(c_data.min(),        c_data.max()))
    print('pressure_1   - min: {:.2f} | max: {:.2f}'.format(y1_data[...,0].min(), y1_data[...,0].max()))
    print('saturation_1 - min: {:.2f} | max: {:.2f}'.format(y1_data[...,-1].min(), y2_data[...,-1].max()))
    print('saturation_2 - min: {:.2f} | max: {:.2f}'.format(y2_data[...,-1].min(), y2_data[...,-1].max()))

    datas = {'X': X_data, 'c': c_data, 'y1': y1_data, 'y2': y2_data}
    norms = {'pmu': pmu, 'psd': psd, 'kmu': kmu, 'ksd': ksd, 'wmi': wmi, 'wma': wma, 
             'tmi': tmi, 'tma': tma, 'vmi': vmi, 'vma': vma, 'cmi': cmi, 'cma': cma}
    return datas, norms