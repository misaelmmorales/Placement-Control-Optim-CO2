import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from sklearn.metrics import r2_score
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIMLoss
from torchmetrics.image import PeakSignalNoiseRatio as PSNRLoss

from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from SlopingAquiferSmall.simulationsVE.convlstm import ConvLSTM

NR, NT = 1272, 40
NX, NY = 40, 40
milli  = 1e-3
mega   = 1e6
Darcy  = 9.869233e-13
psi2pa = 6894.75729
co2rho = 686.5400
sec2yr = 1/(3600*24*365.25)

fno = FNO(n_modes=((10,10,5)), n_layers=4, use_mlp=True,
          in_channels=5, lifting_channels=64, hidden_channels=64, projection_channels=64, out_channels=2)

class Units:
    def __init__(self):
        self.centi = 1e-2
        self.milli = 1e-3
        self.micro = 1e-6
        self.nano = 1e-9
        self.giga = 1e9
        self.mega = 1e6
        self.kilo = 1e3
        self.Darcy = 9.869233e-13
        self.psi2pa = 6894.75729
        self.co2rho = 686.5400
        self.sec2yr = 1/(3600*24*365.25)

##################################################
##################### Models #####################
##################################################
class SqueezeExcite(nn.Module):
    def __init__(self, channels, ratio=4, nonlinearity=F.relu):
        super(SqueezeExcite, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite1 = nn.Linear(channels, channels//ratio)
        self.excite2 = nn.Linear(channels//ratio, channels)
        self.act1    = nonlinearity
        self.act2    = nn.Sigmoid()

    def forward(self, inputs):
        x = self.squeeze(inputs)
        x = x.view(x.size(0), -1)
        x = self.act1(self.excite1(x))
        x = self.act2(self.excite2(x))
        x = x.view(x.size(0), x.size(1), 1, 1)
        s = torch.mul(inputs, x)
        a = torch.add(inputs, s)
        return a
    
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, spectral:bool=True, nonlinearity=F.gelu,
                 n_modes=(10,10), num_layers=1, kernel_size=(3,3), stride=1, padding=1, device='cpu'):
        super(EncoderLayer, self).__init__()
        if spectral:
            self.conv = SpectralConv(in_channels, out_channels, n_modes, n_layers=num_layers).to(device)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels)
        self.saex = SqueezeExcite(out_channels)
        self.norm = nn.GroupNorm(out_channels, out_channels, device=device)
        self.actv = nn.PReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.saex(x)
        x = self.norm(x)
        x = self.actv(x)
        x = self.pool(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, residual=None, spectral:bool=True, 
                 n_modes=(10,10,5), kernel_size:list=[3,3], stride=1, padding=1, num_layers=1, nonlinearity=F.gelu,
                 batch_first=True, bias=True, return_all_layers=False, device='cpu'):
        
        super(DecoderLayer, self).__init__()
        self.convlstm = ConvLSTM(input_dim=in_channels, hidden_dim=out_channels, spectral=spectral, kernel_size=tuple(kernel_size), num_layers=num_layers, batch_first=batch_first, bias=bias, return_all_layers=return_all_layers, device=device)
        self.norm     = nn.GroupNorm(out_channels, out_channels, device=device)
        self.actv     = nn.PReLU(device=device)
        if spectral:
            self.upsm = nn.Upsample(scale_factor=(2,2,1))
            self.conv = SpectralConv(out_channels, out_channels, n_modes=n_modes, n_layers=num_layers).to(device)
        else:
            self.upsm = nn.ConvTranspose3d(out_channels, out_channels, tuple(kernel_size+[2]), stride+1, padding, output_padding=padding, groups=out_channels, device=device)
            self.conv = nn.Conv3d(out_channels, out_channels, tuple(kernel_size+[3]), stride, padding, groups=out_channels, device=device)
        self.residual = residual

    def forward(self, x):
        w, _ = self.convlstm(x)
        x = w[0].permute(0,2,3,4,1) # (b,t,c,h,w) -> (b,c,h,w,t)
        x = self.norm(x)
        x = self.actv(x)
        x = self.upsm(x)
        if self.residual is not None:
            r = self.residual.permute(0,2,3,4,1)
            x = torch.einsum('bchwt,bchwt->bchwt', x, r)
            x = self.conv(x)
            x = self.norm(x)
            x = self.actv(x)
        return x.permute(0,4,1,2,3) # (b,c,h,w,t) -> (b,t,c,h,w)

class Pix2Vid(nn.Module):
    def __init__(self, spectral:bool=True,
                 in_channels:int=4, out_channels_1:int=2, out_channels_2:int=1, c_channels:int=5, 
                 n_timesteps:int=NT//2, c_nonlinearity=F.relu, hidden_channels=[16,64,256], device='cpu'):
        super(Pix2Vid, self).__init__()
        self.enc1 = EncoderLayer(in_channels,        hidden_channels[0], device=device, spectral=spectral)
        self.enc2 = EncoderLayer(hidden_channels[0], hidden_channels[1], device=device, spectral=spectral)
        self.enc3 = EncoderLayer(hidden_channels[1], hidden_channels[2], device=device, spectral=spectral)
        self.mon1 = EncoderLayer(1,                  hidden_channels[0], device=device, spectral=spectral)
        self.mon2 = EncoderLayer(hidden_channels[0], hidden_channels[1], device=device, spectral=spectral)
        self.mon3 = EncoderLayer(hidden_channels[1], hidden_channels[2], device=device, spectral=spectral)
        self.lift = nn.Linear(c_channels, hidden_channels[2])
        self.out1 = nn.Linear(in_channels, out_channels_1)
        self.out2 = nn.Linear(in_channels, out_channels_2)
        self.cact = c_nonlinearity
        self.nt   = n_timesteps
        self.device = device
        self.hidden_channels = hidden_channels
        self.spectral = spectral

    def cond_decoder_layer(self, inputs, controls, residuals, previous_step):
        hidden    = self.hidden_channels
        spectral  = self.spectral
        # conditional
        c = controls.view(controls.size(0), 1, controls.size(-1), 1, 1)
        x = torch.einsum('btchw, bkcpq -> btchw', inputs, c)
        # spatiotemporal
        x = DecoderLayer(hidden[2], hidden[1], residuals[0], device=self.device, spectral=spectral)(x)
        x = DecoderLayer(hidden[1], hidden[0], residuals[1], device=self.device, spectral=spectral)(x)
        x = DecoderLayer(hidden[0], 4, None, device=self.device)(x)
        if previous_step is not None:
            x = torch.cat([x, previous_step], dim=1)
        return x
    
    def uncond_decoder_layer(self, inputs, residuals, previous_step):
        hidden   = self.hidden_channels
        spectral = self.spectral
        # spatiotemporal
        x = DecoderLayer(hidden[2], hidden[1], residuals[0], device=self.device, spectral=spectral)(inputs)
        x = DecoderLayer(hidden[1], hidden[0], residuals[1], device=self.device, spectral=spectral)(x)
        x = DecoderLayer(hidden[0], 4, None, device=self.device)(x)
        if previous_step is not None:
            x = torch.cat([x, previous_step], dim=1)
        return x
    
    def forward(self, x, c):
        # Encoder (spatial)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        z1 = x1.unsqueeze(1)
        z2 = x2.unsqueeze(1)
        z3 = x3.unsqueeze(1)
        # Decoder (injection)
        zc = self.cact(self.lift(c))
        for t in range(self.nt):
            c = zc[:,t]
            if t == 0:
                y = self.cond_decoder_layer(z3, c, [z2, z1], None)
            else:
                y = self.cond_decoder_layer(z3, c, [z2, z1], y)
        y1 = self.out1(y.permute(0,1,3,4,2)).permute(0,1,4,2,3)

        # Encoder (post-injection)
        yy = y1[:,-1,-1].unsqueeze(1)
        u1 = self.mon1(yy)
        u2 = self.mon2(u1)
        u3 = self.mon3(u2)
        w1 = u1.unsqueeze(1)
        w2 = u2.unsqueeze(1)
        w3 = u3.unsqueeze(1)
        # Decoder (monitor)
        for t in range(self.nt):
            if t == 0:
                y = self.uncond_decoder_layer(w3, [w2, w1], None)
            else:
                y = self.uncond_decoder_layer(w3, [w2, w1], y)
        y2 = self.out2(y.permute(0,1,3,4,2)).permute(0,1,4,2,3)
        return y1, y2

##################################################
##################### Losses #####################
##################################################
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.80, gamma=0.80):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.psnr = PSNRLoss()

    def forward(self, true, pred):
        mse_loss = self.mse(true, pred)
        mae_loss = self.mae(true, pred)
        reconstruction = self.beta * mse_loss + (1-self.beta) * mae_loss

        ssim_loss = 1 - self.ssim(true, pred)
        psnr_loss = 1 / self.psnr(true, pred)
        perceptual = self.gamma * ssim_loss + (1-self.gamma) * psnr_loss
        
        return self.alpha * reconstruction + (1-self.alpha) * perceptual
    
class DualCustomLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.80, gamma=0.80):
        super(DualCustomLoss, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.mseloss = nn.MSELoss()
        self.maeloss = nn.L1Loss()
        self.ssimloss = SSIMLoss()
        self.psnrloss = PSNRLoss()

    def forward(self, y1_true, y2_true, y1_pred, y2_pred):
        # injection period
        mse1 = self.mseloss(y1_pred, y1_true)
        mae1 = self.maeloss(y1_pred, y1_true)
        reconstruction1 = self.beta * mse1 + (1 - self.beta) * mae1

        p1 = y1_true.permute(0,2,3,4,1)
        q1 = y1_pred.permute(0,2,3,4,1)
        ssim1 = 1 - self.ssimloss(p1, q1)
        psnr1 = 1 / self.psnrloss(p1, q1)
        perceptual1 = self.gamma * ssim1 + (1 - self.gamma) * psnr1

        # monitor period
        mse2 = self.mseloss(y2_pred, y2_true)
        mae2 = self.maeloss(y2_pred, y2_true)
        reconstruction2 = self.beta * mse2 + (1 - self.beta) * mae2

        p2 = y2_true.permute(0,2,3,4,1)
        q2 = y2_pred.permute(0,2,3,4,1)
        ssim2 = 1 - self.ssimloss(p2, q2)
        psnr2 = 1 / self.psnrloss(p2, q2)
        perceptual2 = self.gamma * ssim2 + (1 - self.gamma) * psnr2

        loss1 = self.alpha * reconstruction1 + (1 - self.alpha) * perceptual1
        loss2 = self.alpha * reconstruction2 + (1 - self.alpha) * perceptual2
        return (loss1 + loss2) / 2
    
class LpLoss(nn.Module):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction    = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms   = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms      = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def forward(self, x, y):
        return self.rel(x, y)
    
class DualLpLoss(nn.Module):
    def __init__(self, alpha=0.5, d=2, p=2, size_average=True, reduction=True):
        super(DualLpLoss, self).__init__()
        self.loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)
        self.alpha = alpha

    def forward(self, y1_true, y2_true, y1_pred, y2_pred):
        loss1 = self.loss(y1_true, y1_pred)
        loss2 = self.loss(y2_true, y2_pred)
        return self.alpha * loss1 + (1-self.alpha) * loss2

##################################################
################### DataLoader ###################
##################################################
def fno_dataset():
    # Load volumes
    v = np.load('volumes.npz')
    conversion = co2rho / 1e3 / mega
    freeVol    = v['freeVol']    * conversion
    trappedVol = v['trappedVol'] * conversion
    leakedVol  = v['leakedVol']  * conversion
    totVol     = v['totVol']     * conversion
    v.close()
    all_volumes = {'free': freeVol, 'trapped': trappedVol, 'leaked': leakedVol, 'total': totVol}

    # Load data
    X_data  = np.load('X_data.npy')
    c_data  = np.load('c_data.npy') * co2rho
    y1_data = np.load('y1_data.npy')
    y2_data = np.load('y2_data.npy')

    # Normalize data
    X_data[...,0] = X_data[...,0] / 0.27
    X_data[...,1] = X_data[...,1] / 3.3
    X_data[...,2] = (X_data[...,2]-900) / (1042 - 900)
    c_data = c_data / 10
    y1_data[...,0] = y1_data[...,0] / 1e4

    # Expand data
    controls = np.zeros((NR,30,64,64,1), dtype=np.float32)
    for i in tqdm(range(NR)):
        w = np.argwhere(X_data[i,...,3])
        nw = w.shape[0]
        for j in range(30):
            for k in range(nw):
                controls[i, j, w[k,0], w[k,1]] = c_data[i,j,k]
    X_aug = np.concatenate([np.repeat(np.expand_dims(X_data, axis=1), 30, axis=1), controls], axis=-1)
    print('X: {} | y1: {} | y2: {}'.format(X_aug.shape, y1_data.shape, y2_data.shape))

    # Reshape data
    X_aug   = np.moveaxis(np.moveaxis(X_aug,   -1, 1), 2, -1)
    y1_data = np.moveaxis(np.moveaxis(y1_data, -1, 1), 2, -1)
    y2_data = np.moveaxis(np.moveaxis(y2_data, -1, 1), 2, -1)

    # Tensorize data
    X_data  = torch.tensor(X_aug, dtype=torch.float32)
    y1_data = torch.tensor(y1_data, dtype=torch.float32)
    y2_data = torch.tensor(y2_data, dtype=torch.float32)
    print('X: {} | y1: {} | y2: {}'.format(X_data.shape, y1_data.shape, y2_data.shape))

    # Split and DataLoader
    idx = np.random.choice(range(NR), NR, replace=False)
    train_idx, valid_idx, test_idx= idx[:1000], idx[1000:1136], idx[1136:]

    X_train, y1_train, y2_train = X_data[train_idx], y1_data[train_idx], y2_data[train_idx]
    print('Train - n: {} | X: {} | y1: {} | y2: {}'.format(len(X_train), X_train.shape, y1_train.shape, y2_train.shape))

    X_valid, y1_valid, y2_valid = X_data[valid_idx], y1_data[valid_idx], y2_data[valid_idx]
    print('Valid - n: {}  | X: {}  | y1: {}  | y2: {}'.format(len(X_valid), X_valid.shape, y1_valid.shape, y2_valid.shape))

    X_test, y1_test, y2_test = X_data[test_idx], y1_data[test_idx], y2_data[test_idx]
    print('Test -  n: {}  | X: {}  | y1: {}  | y2: {}'.format(len(X_test), X_test.shape, y1_test.shape, y2_test.shape))

    trainloader = DataLoader(TensorDataset(X_train, y1_train), batch_size=8, shuffle=True)
    validloader = DataLoader(TensorDataset(X_valid, y1_valid), batch_size=8, shuffle=False)

    return (X_data, y1_data, y2_data, all_volumes, idx), (trainloader, validloader)

def pix2vid_dataset(folder:str='simulations_40x40', idx=None, batch_size:int=8, 
                    normalize:bool=True, tensorize:bool=True, send_to_device:bool=False, device=None):
    # Load volumes
    v = np.load('{}/volumes.npz'.format(folder))
    conversion = co2rho / 1e3 / mega
    freeVol    = v['freeVol']    * conversion
    trappedVol = v['trappedVol'] * conversion
    leakedVol  = v['leakedVol']  * conversion
    totVol     = v['totVol']     * conversion
    v.close()
    all_volumes = {'free': freeVol, 'trapped': trappedVol, 'leaked': leakedVol, 'total': totVol}

    # Load data
    X_data  = np.load('{}/X_data.npy'.format(folder))
    c_data  = np.load('{}/c_data.npy'.format(folder))
    y1_data = np.load('{}/y1_data.npy'.format(folder))
    y2_data = np.load('{}/y2_data.npy'.format(folder))
    print('X: {} | c: {} | y1: {} | y2: {}'.format(X_data.shape, c_data.shape, y1_data.shape, y2_data.shape))

    # Normalize data
    if normalize:
        X_data[:,0] = X_data[:,0] / 0.37
        X_data[:,1] = X_data[:,1] / 3.4
        X_data[:,2] = (X_data[:,2] - X_data[:,2].min()) / (X_data[:,2].max() - X_data[:,2].min())
        c_data = c_data / 7
        y1_data[:,:,0] = y1_data[:,:,0] / 1e4

    # Tensorize data
    if tensorize:
        Xt = torch.tensor(X_data, dtype=torch.float32)
        ct = torch.tensor(c_data, dtype=torch.float32)
        y1t = torch.tensor(y1_data, dtype=torch.float32)
        y2t = torch.tensor(y2_data, dtype=torch.float32)

    # Send to device
    if send_to_device == True:
        assert tensorize is True, 'Please tensorize the data first'
        assert device is not None, 'Please provide a device'
        Xt = Xt.to(device)
        ct = ct.to(device)
        y1t = y1t.to(device)
        y2t = y2t.to(device)

    # Split and DataLoader
    if idx is None:
        idx = np.random.choice(range(NR), NR, replace=False)
        np.save('training_idx.npy', idx)
    else:
        idx = np.load('training_idx.npy')
    train_idx, valid_idx, test_idx= idx[:1000], idx[1000:1136], idx[1136:]
    X_train, c_train, y1_train, y2_train = Xt[train_idx], ct[train_idx], y1t[train_idx], y2t[train_idx]
    X_valid, c_valid, y1_valid, y2_valid = Xt[valid_idx], ct[valid_idx], y1t[valid_idx], y2t[valid_idx]
    X_test,  c_test,  y1_test,  y2_test  = Xt[test_idx],  ct[test_idx],  y1t[test_idx],  y2t[test_idx]
    print('-'*100)
    print('Train - X:  {}     | c:  {}'.format(X_train.shape, c_train.shape))
    print('        y1: {} | y2: {}'.format(y1_train.shape, y2_train.shape))
    print('-'*20)
    print('Valid - X:  {}     | c:  {}'.format(X_valid.shape, c_valid.shape))
    print('        y1: {} | y2: {}'.format(y1_valid.shape, y2_valid.shape))
    print('-'*20)
    print('Test  - X:  {}     | c:  {}'.format(X_test.shape, c_test.shape))
    print('        y1: {} | y2: {}'.format(y1_test.shape, y2_test.shape))

    trainloader = DataLoader(TensorDataset(X_train, c_train, y1_train, y2_train), batch_size=batch_size, shuffle=True)
    validloader = DataLoader(TensorDataset(X_valid, c_valid, y1_valid, y2_valid), batch_size=batch_size, shuffle=False)

    return (Xt, ct, y1t, y2t, all_volumes, idx), (trainloader, validloader)

##################################################
################### Utilities ####################
##################################################
def check_torch(verbose:bool=True):
    if torch.cuda.is_available():
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('Torch device: {}'.format(device))
            print('-'*60)
        return device
    else:
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        device = torch.device('cpu')
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('Torch device: {}'.format(device))
            print('-'*60)
        return device
    
def calculate_metrics(y1, y2, u1, u2, data_range=1):
    p1, s1, s2 = y1[:,:,0], y1[:,:,1], y2[:,:,-1]
    q1, z1, z2 = u1[:,:,0], u1[:,:,1], u2[:,:,-1]
    r2p = r2_score(p1.reshape(p1.shape[0],-1), q1.reshape(q1.shape[0],-1))
    r2s = r2_score(s1.reshape(s1.shape[0],-1), z1.reshape(z1.shape[0],-1))
    r2m = r2_score(s2.reshape(s2.shape[0],-1), z2.reshape(z2.shape[0],-1))
    mse_p = mean_squared_error(p1, q1)
    mse_s = mean_squared_error(s1, z1)
    mse_m = mean_squared_error(s2, z2)
    ssim_p = structural_similarity(p1, q1, data_range=data_range)
    ssim_s = structural_similarity(s1, z1, data_range=data_range)
    ssim_m = structural_similarity(s2, z2, data_range=data_range)
    psnr_p = peak_signal_noise_ratio(p1, q1, data_range=data_range)
    psnr_s = peak_signal_noise_ratio(s1, z1, data_range=data_range)
    psnr_m = peak_signal_noise_ratio(s2, z2, data_range=data_range)
    print('-'*81+'\n'+'-'*36+' METRICS '+'-'*36+'\n'+'-'*81)
    print('R2   - pressure: {:.4f} | saturation (inj): {:.4f} | saturation (monitor): {:.4f}'.format(r2p, r2s, r2m))
    print('MSE  - pressure: {:.4f} | saturation (inj): {:.4f} | saturation (monitor): {:.4f}'.format(mse_p, mse_s, mse_m))
    print('SSIM - pressure: {:.4f} | saturation (inj): {:.4f} | saturation (monitor): {:.4f}'.format(ssim_p, ssim_s, ssim_m))
    print('PSNR - pressure: {:.4f} | saturation (inj): {:.4f} | saturation (monitor): {:.4f}'.format(psnr_p, psnr_s, psnr_m))
    print('-'*81)

