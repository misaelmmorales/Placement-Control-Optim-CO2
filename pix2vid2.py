import keras
import tensorflow as tf
from keras import Model, layers, callbacks, losses

NX = 64
NY = 64
NZ = 1
NTT = 40
NT1 = 20
NT2 = 10 #5
X_CHANNELS = 5
Y1_CHANNELS = 2
Y2_CHANNELS = 1
DROPOUT = 0.05

def encoder_layer(inp, filt, k=3, pad='same', pool=(2,2), drop=DROPOUT, lambda_reg=1e-6):
    def SqueezeExcite2d(z, ratio:int=4):
        _ = layers.GlobalAveragePooling2D()(z)
        _ = layers.Dense(filt//ratio, activation='gelu')(_)
        _ = layers.Dense(filt, activation='sigmoid')(_)
        _ = layers.Reshape((1, 1, filt))(_)
        w = layers.Multiply()([z, _])
        return layers.Add()([z, w])
    _ = layers.SeparableConv2D(filt, k, padding=pad)(inp)
    _ = SqueezeExcite2d(_)
    _ = layers.GroupNormalization(groups=-1)(_)
    _ = layers.PReLU()(_)
    _ = layers.MaxPooling2D(pool)(_)
    #_ = layers.SpatialDropout2D(drop)(_)
    return _

def lifting_layer(inp, dim, drop=DROPOUT, nonlinearity='relu'):
    _ = layers.Dense(dim)(inp)
    #_ = layers.Dropout(drop)(_)
    _ = layers.PReLU()(_)
    return _

def lifting_squeezexcite_layer(inp, dim, drop=DROPOUT, nonlinearity='relu'):
    def SqueezeExcite1d(z, ratio:int=4):
        _ = layers.GlobalAveragePooling1D()(z)
        _ = layers.Dense(dim//ratio, activation='relu')(_)
        _ = layers.Dense(dim, activation='sigmoid')(_)
        _ = layers.Reshape((1, 1, dim))(_)
        w = layers.Multiply()([z, _])
        return layers.Add()([z, w])
    _ = layers.Dense(dim)(inp)
    #_ = layers.Dropout(drop)(_)
    _ = layers.PReLU()(_)
    _ = SqueezeExcite1d(_)
    return _

def lifting_attention_layer(inp, dim, nheads:int=4, drop=DROPOUT, nonlinearity='relu'):
    _ = layers.Dense(dim)(inp)
    #_ = layers.Dropout(drop)(_)
    _ = layers.PReLU()(_)
    _, a = layers.MultiHeadAttention(nheads, key_dim=NT1, value_dim=NT1)(_, _, return_attention_scores=True)
    return _, a

def recurrent_step(inp, filt, res, kern=3, pad='same', drop=DROPOUT, leaky_slope=0.3):
    y = layers.ConvLSTM2D(filt, kern, padding=pad)(inp)
    y = layers.GroupNormalization(groups=-1)(y)
    y = layers.PReLU()(y)
    y = layers.Conv2DTranspose(filt, kern, padding=pad, strides=2)(y)
    #y = layers.SpatialDropout2D(drop)(y)
    y = layers.Concatenate()([y, res])
    y = layers.Conv2D(filt, kern, padding=pad)(y)
    #y = layers.Activation('sigmoid')(y)
    y = layers.PReLU()(y)
    _, h, w, c = y.shape
    y = layers.Reshape((1, h, w, c))(y)
    return y

def recurrent_last(inp, filt, kern=3, pad='same', drop=DROPOUT, leaky_slope=0.3, out_channels=2):
    y = layers.ConvLSTM2D(filt, kern, padding=pad)(inp)
    y = layers.GroupNormalization(groups=-1)(y)
    y = layers.PReLU()(y)
    y = layers.Conv2DTranspose(filt, kern, padding=pad, strides=2)(y)
    #y = layers.SpatialDropout2D(drop)(y)
    y = layers.Conv2D(out_channels, kern, padding=pad)(y)
    #y = layers.Activation('sigmoid')(y)
    y = layers.PReLU()(y)
    _, h, w, c = y.shape
    y = layers.Reshape((1, h, w, c))(y)
    return y

def conditional_recurrent_decoder(z_input, c_input, residuals, rnn_filters=[8,16,64], 
                                  previous_timestep=None, leaky_slope=0.3, out_channels:int=Y1_CHANNELS):
    h, w, c = z_input.shape[1:]
    zz = layers.Reshape((1, h, w, c))(z_input)
    zc = layers.Reshape((1, c))(c_input)
    _ = layers.Multiply()([zz, zc])
    _ = recurrent_step(_, rnn_filters[0], residuals[0], leaky_slope=leaky_slope)
    _ = recurrent_step(_, rnn_filters[1], residuals[1], leaky_slope=leaky_slope)
    _ = recurrent_last(_, rnn_filters[2], leaky_slope=leaky_slope, out_channels=out_channels)
    if previous_timestep is not None:
        _ = layers.Concatenate(axis=1)([previous_timestep, _])
    return _

def unconditional_recurrent_decoder(z_input, residuals, rnn_filters=[8,16,64], 
                                    previous_timestep=None, leaky_slope=0.3, out_channels:int=Y2_CHANNELS):    
    h, w, c = z_input.shape[1:]
    _ = layers.Reshape((1, h, w, c))(z_input)
    _ = recurrent_step(_, rnn_filters[0], residuals[0], leaky_slope=leaky_slope)
    _ = recurrent_step(_, rnn_filters[1], residuals[1], leaky_slope=leaky_slope)
    _ = recurrent_last(_, rnn_filters[2], leaky_slope=leaky_slope, out_channels=out_channels)
    if previous_timestep is not None:
        _ = layers.Concatenate(axis=1)([previous_timestep, _])
    return _

def make_model(hidden=[8, 16, 64], verbose:bool=True):  
    x_inp = layers.Input(shape=(NX, NY, X_CHANNELS))
    c_inp = layers.Input(shape=(NT1, 5))

    x1 = encoder_layer(x_inp, hidden[0])
    x2 = encoder_layer(x1, hidden[1])
    x3 = encoder_layer(x2, hidden[2])
    zc, ac = lifting_attention_layer(c_inp, hidden[2])
    #zc = lifting_squeezexcite_layer(c_inp, hidden[2])
    #zc = lifting_layer(c_inp, hidden[2])
    t1 = None
    for t in range(NT1):
        if t==0:
            t1 = conditional_recurrent_decoder(x3, zc[...,t,:], [x2, x1], rnn_filters=hidden)
        else:
            t1 = conditional_recurrent_decoder(x3, zc[...,t,:], [x2, x1], rnn_filters=hidden, previous_timestep=t1) 
    t1 = layers.TimeDistributed(layers.SeparableConv2D(Y1_CHANNELS, 3, activation='relu', padding='same'))(t1)

    t2 = None
    for t in range(NT2):
        if t==0:
            t2 = conditional_recurrent_decoder(x3, zc[...,t,:], [x2, x1], rnn_filters=hidden, out_channels=Y2_CHANNELS)
        else:
            t2 = conditional_recurrent_decoder(x3, zc[...,t,:], [x2, x1], rnn_filters=hidden, out_channels=Y2_CHANNELS, previous_timestep=t2)
    t2 = layers.TimeDistributed(layers.Conv2D(Y2_CHANNELS, 3, activation='sigmoid', padding='same'))(t2)

    model = Model(inputs=[x_inp, c_inp], outputs=[t1, t2])
    if verbose: print('# parameters: {:,}'.format(model.count_params()))
    return model

@keras.saving.register_keras_serializable()
class CustomLoss(losses.Loss):
    def __init__(self, a=0.8, b=0.8, name='custom_loss'):
        super(CustomLoss, self).__init__(name=name)
        self.a = a
        self.b = b
        self.mse = losses.MeanSquaredError()
        self.mae = losses.MeanAbsoluteError()

    def call(self, true, pred):
        ssim_loss  = 1.0 - tf.reduce_mean(tf.image.ssim(true, pred, max_val=1.0))
        mse_loss   = self.mse(true, pred)
        mae_loss   = self.mae(true, pred)
        pixel_loss = self.b * mse_loss + (1 - self.b) * mae_loss
        return self.a * pixel_loss + (1 - self.a) * ssim_loss

class MonitorCallback(callbacks.Callback):
    def __init__(self, monitor:int=10, verbose:int=1):
        super(MonitorCallback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.monitor == 0:
            if self.verbose == 2:
                print('Epoch: {} | Loss: {:.5f} | Val Loss: {:.5f}'.format(epoch+1, logs['loss'], logs['val_loss']))
            elif self.verbose == 1:
                print('Epoch: {} | Loss: {:.5f}'.format(epoch+1, logs['loss']))
            elif self.verbose == 0:
                pass
            else:
                raise ValueError('Invalid verbose value. Use 0, 1 or 2.')