import os
from time import time
import numpy as np
import pandas as pd
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
from neuralop.models import *
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from transformers import Swinv2Model, Swinv2Config

sec2year   = 365.25 * 24 * 60 * 60
psi2pascal = 6894.76
co2_rho    = 686.5266
mega       = 1e6

n_timesteps = 33
nx, ny, nz  = 100, 100, 11

indexMap = loadmat('data_100_100_11/G_cells_indexMap.mat', simplify_cells=True)['gci']
Grid = np.zeros((nx,ny,nz)).flatten(order='F')
Grid[indexMap] = 1
Grid = Grid.reshape(nx,ny,nz, order='F')
Tops = np.load('data_npy_100_100_11/tops_grid.npz')['tops']

def check_torch(verbose:bool=True):
    if torch.cuda.is_available():
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    else:
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('-'*60)
        device = torch.device('cpu')
        return device

device = check_torch()

class FTMiONet(nn.Module):
    def __init__(self, hidden_1:int=16, hidden_2:int=32, hidden_3:int=64):
        super(FTMiONet, self).__init__()
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3

        self.conv1 = nn.Conv3d(2, hidden_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_1, hidden_2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(hidden_2, hidden_3, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(hidden_1)
        self.norm2 = nn.InstanceNorm3d(hidden_2)
        self.norm3 = nn.InstanceNorm3d(hidden_3)
        self.pool = nn.MaxPool3d((1,1,2))
        self.gelu = nn.GELU()

        self.vitm = Swinv2Model(Swinv2Config(image_size=100, num_channels=hidden_3,
                                             embed_dim=96, num_heads=[2,4,8,16],
                                             output_attention=True,
                                             output_hidden_states=True))

        self.vitg = Swinv2Model(Swinv2Config(image_size=100, num_channels=hidden_3,
                                             embed_dim=96, num_heads=[2,4,8,16],
                                             output_attention=True,
                                             output_hidden_states=True))

        self.tel1 = nn.TransformerEncoderLayer(d_model=10, nhead=2, dim_feedforward=1024, activation='gelu', batch_first=True)
        self.tel2 = nn.TransformerEncoderLayer(d_model=160, nhead=8, dim_feedforward=1024, activation='gelu', batch_first=True)
        self.tel3 = nn.TransformerEncoderLayer(d_model=32, nhead=8, dim_feedforward=1024, activation='gelu', batch_first=True)

        self.trf1 = nn.TransformerEncoder(self.tel1, num_layers=4)
        self.trf2 = nn.TransformerEncoder(self.tel2, num_layers=4)
        self.trf3 = nn.TransformerEncoder(self.tel3, num_layers=4)

        self.fno = FNO(n_modes=(1,4), n_layers=2, norm='instance_norm',
                       in_channels=2, 
                       lifting_channels=hidden_1, 
                       hidden_channels=hidden_3, 
                       projection_channels=hidden_1,
                       out_channels=2)
        self.lift = nn.Linear(1920, 29128)

    def forward(self, x):
        xm, xg, xw, xc, xt = x

        zm = self.pool(self.norm1(self.gelu(self.conv1(xm))))
        zm = self.pool(self.norm2(self.gelu(self.conv2(zm))))
        zm = self.pool(self.norm3(self.gelu(self.conv3(zm)))).squeeze()
        mv = self.vitm(zm)
        zm = mv['reshaped_hidden_states'][-1].reshape(zm.shape[0], -1)

        zg = self.pool(self.norm1(self.gelu(self.conv1(xg))))
        zg = self.pool(self.norm2(self.gelu(self.conv2(zg))))
        zg = self.pool(self.norm3(self.gelu(self.conv3(zg)))).squeeze()
        gv = self.vitg(zg)
        zg = gv['reshaped_hidden_states'][-1].reshape(zg.shape[0], -1)

        zw = xw.view(xw.shape[0], -1)
        zw = self.trf1(zw)

        zc = xc.view(xc.shape[0], -1)
        zc = self.trf2(zc)

        zt = xt.view(xt.shape[0], -1)
        zt = self.trf3(zt)

        mg = torch.einsum('bp,bp->bp', zm, zg)
        wc = torch.einsum('bw,bc->bwc', zw, zc)
        zb = torch.einsum('bp,bwc->bwcp', mg, wc)
        zb = zb.reshape(zb.shape[0], 2, 5, 5, 32, 32, 384)
        zb = torch.einsum('blwwttp,blwwttp->blwtp', zb, zb)
        merge = torch.einsum('blwtp,bt->blwtp', zb, zt)
        merge = merge.permute(0,1,3,2,4).reshape(merge.shape[0], 2, 32, -1)
        zy = self.fno(merge)
        yy = self.lift(zy)
        
        return yy
    
class CustomDataset(Dataset):
    def __init__(self, data_folder:str='data_npy_100_100_11'):
        self.data_folder = data_folder
        
        self.x_folder = os.path.join(data_folder, 'inputs_rock_rates_locs_time')
        self.y_folder = os.path.join(data_folder, 'outputs_masked_pressure_saturation')

        self.x_file_list = os.listdir(self.x_folder)
        self.y_file_list = os.listdir(self.y_folder)

    def __len__(self):
        return len(self.x_file_list)
    
    def __getitem__(self, idx):
        x  = np.load(os.path.join(self.x_folder, self.x_file_list[idx]))
        y  = np.load(os.path.join(self.y_folder, self.y_file_list[idx]))

        xg = np.concatenate([np.expand_dims(Tops/(3000), 0), 
                             np.expand_dims(Grid, 0)], 
                             axis=0)

        xm = np.concatenate([np.expand_dims(x['poro']/(0.3),0), 
                             np.expand_dims(x['perm']/(3.3),0)], 
                             axis=0)
        
        xw = x['locs']           / (100)
        xc = x['ctrl'][1:]       * co2_rho*sec2year/mega/1e3/(25)
        xt = x['time'][1:]       / sec2year / (100)
        yp = y['pressure'][2:]   / psi2pascal / (1e4)
        ys = y['saturation'][2:] / 0.8
        yy = np.concatenate([np.expand_dims(yp,0), np.expand_dims(ys,0)], axis=0)

        xm = torch.tensor(xm, dtype=torch.float32, device=device)
        xg = torch.tensor(xg, dtype=torch.float32, device=device)
        xw = torch.tensor(xw, dtype=torch.float32, device=device)
        xc = torch.tensor(xc, dtype=torch.float32, device=device)
        xt = torch.tensor(xt, dtype=torch.float32, device=device)
        yy = torch.tensor(yy, dtype=torch.float32, device=device)

        return (xm, xg, xw, xc, xt), yy
    
dataset = CustomDataset()
trainset, testset  = random_split(dataset,  [1172, 100])
trainset, validset = random_split(trainset, [972,  200])

trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
validloader = DataLoader(validset, batch_size=16, shuffle=False)
testloader  = DataLoader(testset, batch_size=16, shuffle=False)

model = FTMiONet().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

print('# params: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

epochs, monitor = 301, 50
train_loss, valid_loss = [], []
start = time()
for epoch in range(epochs):
    # training
    epoch_train_loss = []
    model.train()
    for i, (x,y) in enumerate(trainloader):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())
    train_loss.append(np.mean(epoch_train_loss))
    # validation
    epoch_valid_loss = []
    model.eval()
    with torch.no_grad():
        for i, (xv,yv) in enumerate(validloader):
            yvhat = model(xv)
            loss = criterion(yvhat, yv)
            epoch_valid_loss.append(loss.item())
    valid_loss.append(np.mean(epoch_valid_loss))
    # monitor
    if epoch % monitor == 0:
        print('Epoch: {}/{} | Loss: {:.4f} | Valid Loss: {:.4f}'.format(epoch+1, epochs-1, train_loss[-1], valid_loss[-1]))
train_time = time() - start
print('Total training time: {:.3f} minutes'.format(train_time/60))

torch.save(model.state_dict(), 'FTMiONet.pth')
losses = pd.DataFrame({'train': train_loss, 'valid': valid_loss})
losses.to_csv('FTMiONet_losses.csv', index=False)

for i, (xx,yy) in enumerate(dataset):
    yp = model(xx).detach().cpu().numpy()
    np.save('data_npy_100_100_11/predictions_ftmionet/y_pred_{}'.format(i), yp)
print('Predictions saved!')