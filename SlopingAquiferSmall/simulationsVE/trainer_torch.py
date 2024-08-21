import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.io as sio
from time import time

import torch
import torch.optim as optim
from utils import Units, check_torch, pix2vid_dataset
from utils import Pix2Vid, DualCustomLoss, DualLpLoss

NR, NT = 1272, 40
NX, NY = 40, 40
units  = Units()
folder = 'simulations_40x40'
device = check_torch()

tt = np.load('{}/data/timesteps.npz'.format(folder))
timesteps, deltaTime = tt['timesteps'], tt['deltatime']
t0steps = timesteps[:20]
print('timesteps: {} | deltaT: {}'.format(len(timesteps), np.unique(deltaTime)))

tops2d = sio.loadmat('{}/grids/Gt.mat'.format(folder), simplify_cells=True)['Gt']['cells']['z'].reshape(NX,NY,order='F')
print('tops2d: {}'.format(tops2d.shape))

(Xt, ct, y1t, y2t, all_volumes, idx), (trainloader, validloader) = pix2vid_dataset(folder='simulations_40x40/data',
                                                                                   batch_size=32,
                                                                                   send_to_device=True,
                                                                                   device=device)

model = Pix2Vid(device=device, spectral=False).to(device)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('# parameters: {:,} | device: {}'.format(nparams, model.device))
criterion = DualCustomLoss(gamma=1.0).to(device) # DualLpLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

start = time()
epochs, monitor = 200, 5
EarlyStop, tolerance, patience = True, 1e-3, 10
train_loss, valid_loss = [], []
for epoch in range(epochs):
    epoch_train_loss = []
    model.train()
    for i, (x,c,y1,y2) in enumerate(trainloader):
        optimizer.zero_grad()
        u1, u2 = model(x,c)
        loss = criterion(y1, y2, u1, u2)
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())
    train_loss.append(np.mean(epoch_train_loss))
    # validation
    model.eval()
    epoch_valid_loss = []
    with torch.no_grad():
        for i, (x,c,y1,y2) in enumerate(validloader):
            u1, u2 = model(x,c)
            loss = criterion(y1, y2, u1, u2)
            epoch_valid_loss.append(loss.item())
    valid_loss.append(np.mean(epoch_valid_loss))
    # progress
    if (epoch+1) % monitor == 0:
        print('Epoch: [{}/{}] | Train Loss: {:.5f} | Valid Loss: {:.5f}'.format(epoch+1, epochs, train_loss[-1], valid_loss[-1]))
    # early stopping
    if EarlyStop:
        if epoch > patience:
            if np.abs(valid_loss[-1] - valid_loss[-patience]) < tolerance:
                print('Early stopping at epoch: {}'.format(epoch+1))
                break
        

traintime = time()-start
torch.save(model.state_dict(), 'pix2vid_model.pth')
losses = pd.DataFrame({'train': train_loss, 'valid': valid_loss})
losses.to_csv('pix2vid_losses_{:.1f}m.csv'.format(traintime/60), index=False)
print('Total training time: {:.3f} minutes'.format(traintime/60))

plt.figure(figsize=(12,6))
plt.plot(losses.index, losses['train'], label='Train')
plt.plot(losses.index, losses['valid'], label='Valid')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('pix2vid_losses.png')
plt.close()