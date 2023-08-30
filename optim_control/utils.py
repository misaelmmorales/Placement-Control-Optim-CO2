import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage.util import random_noise

import keras.backend as K
from keras import Model, regularizers
from keras.layers import *
from keras.optimizers import SGD, Adam, Nadam
from keras.losses import MeanSquaredError, MeanAbsoluteError

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow_addons.layers import *
from tensorflow_addons.optimizers import AdamW
from tensorflow.image import ssim as SSIM
from tensorflow.keras.metrics import mean_squared_error as MSE
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    cuda_version, cudnn_version = sys_info['cuda_version'], sys_info['cudnn_version']
    num_gpu_avail = len(tf.config.experimental.list_physical_devices('GPU'))
    gpu_name = device_lib.list_local_devices()[1].physical_device_desc[17:40]
    print('... Checking Tensorflow Version ...')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print("TF: {} | CUDA: {} | cuDNN: {}".format(tf.__version__, cuda_version, cudnn_version))
    print('# GPU available: {} ({})'.format(num_gpu_avail, gpu_name))
    #print(tf.config.list_physical_devices())
    return None