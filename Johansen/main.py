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

class MiONet(nn.Module):
    def __init__(self, hidden_channels=16, output_channels=32):
        super(MiONet, self).__init__()
        self.hidden = hidden_channels
        self.output = output_channels

        self.conv1 = nn.Conv3d(2, self.hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(self.hidden, self.output, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(self.hidden)
        self.norm2 = nn.BatchNorm3d(self.output)
        self.pool  = nn.MaxPool3d(2)
        self.gelu  = nn.GELU()

        self.linW1 = nn.Linear(5, self.hidden)
        self.linW2 = nn.Linear(self.hidden, self.output)
        self.bnW1  = nn.BatchNorm1d(self.hidden//8)
        self.bnW2  = nn.BatchNorm1d(self.output//16)

        self.lstmC1 = nn.LSTM(5, self.hidden, num_layers=1, batch_first=True)
        self.lstmC2 = nn.LSTM(self.hidden, self.output, num_layers=1, batch_first=True)

        self.lmstT1 = nn.LSTM(1, self.hidden, num_layers=1, batch_first=True)
        self.lmstT2 = nn.LSTM(self.hidden, self.output, num_layers=1, batch_first=True)

        self.linY1 = nn.Linear(1250, 10000)
        self.linY2 = nn.Linear(10000, 29128)

    def forward(self, x):
        xm, xg, xw, xc, xt = x

        zm = self.gelu(self.pool(self.norm1(self.conv1(xm))))
        zm = self.gelu(self.pool(self.norm2(self.conv2(zm))))
        zm = zm.view(zm.shape[0], self.output, -1)

        zg = self.gelu(self.pool(self.norm1(self.conv1(xg))))
        zg = self.gelu(self.pool(self.norm2(self.conv2(zg))))
        zg = zg.view(zg.shape[0], self.output, -1)

        zw = self.gelu(self.bnW1(self.linW1(xw)))
        zw = self.gelu(self.bnW2(self.linW2(zw)))

        zc, _ = self.lstmC1(xc)
        zc, _ = self.lstmC2(zc)

        zt, _ = self.lmstT1(xt)
        zt, _ = self.lmstT2(zt)

        mg = torch.einsum('bcp,bcp->bcp', zm, zg)
        wc = torch.einsum('blc,btc->btlc', zw, zc)
        branch = torch.einsum('bcp,btlc->btpl', mg, wc)
        merge  = torch.einsum('btpl,btc->btlp', branch, zt)

        yy = self.gelu(self.linY1(merge))
        yy = self.linY2(yy)

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

        xg = np.concatenate([np.expand_dims(Tops/3000, 0), 
                             np.expand_dims(Grid, 0)], 
                             axis=0)

        xm = np.concatenate([np.expand_dims(x['poro']/0.3,0), 
                             np.expand_dims(x['perm']/3.3,0)], 
                             axis=0)
        
        xw = x['locs'] / 100
        xc = np.concatenate([np.zeros((1,xw.shape[-1])), x['ctrl']], axis=0) *co2_rho*sec2year/mega/1e3 /25
        xt = np.expand_dims(np.insert(x['time']/sec2year/100, 0, 0), -1)
        yp = y['pressure'] /psi2pascal/1e4
        ys = y['saturation'] / 0.8
        yy = np.concatenate([np.expand_dims(yp,1), np.expand_dims(ys,1)], axis=1)

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

model = MiONet().to(device)
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

torch.save(model.state_dict(), 'MiONet.pth')
losses = pd.DataFrame({'train': train_loss, 'valid': valid_loss})
losses.to_csv('MiONet_losses.csv', index=False)

for i, (xx,yy) in enumerate(dataset):
    yp = model(xx).detach().cpu().numpy()
    np.save('data_npy_100_100_11/predictions_mionet/y_pred_{}'.format(i), yp)
print('MiONet predictions saved!')