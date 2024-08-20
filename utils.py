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