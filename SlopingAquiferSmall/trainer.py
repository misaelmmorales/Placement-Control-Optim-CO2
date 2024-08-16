import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from utils import Units, check_torch, pix2vid_dataset, DualCustomLoss, DualLpLoss, NeuralPix2Vid

NR, NT = 1272, 40
NX, NY = 40, 40
units  = Units()
folder = 'simulations_40x40'
device = check_torch()

tt = np.load('{}/timesteps.npz'.format(folder))
timesteps, deltaTime = tt['timesteps'], tt['deltatime']
t0steps = timesteps[:20]
print('timesteps: {} | deltaT: {}'.format(len(timesteps), np.unique(deltaTime)))

tops2d = sio.loadmat('{}/Gt.mat'.format(folder), simplify_cells=True)['Gt']['cells']['z'].reshape(NX,NY,order='F')
print('tops2d: {}'.format(tops2d.shape))

(Xt, ct, y1t, y2t, all_volumes, idx), (trainloader, validloader) = pix2vid_dataset(folder='simulations_40x40',
                                                                                   batch_size=32,
                                                                                   send_to_device=True,
                                                                                   device=device)

model = NeuralPix2Vid(device=device).to(device)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('# parameters: {:,} | device: {}'.format(nparams, model.device))
#criterion = DualLpLoss().to(device)
criterion = DualCustomLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

start = time()
epochs, monitor = 200, 5
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

print('Total training time: {:.3f} minuts'.format((time()-start)/60))
torch.save(model.state_dict(), 'neural-pix2vid_model.pth')
losses = pd.DataFrame({'train': train_loss, 'valid': valid_loss})
losses.to_csv('neural-pix2vid_losses.csv', index=False)

plt.figure(figsize=(12,6))
plt.plot(losses.index, losses['train'], label='Train')
plt.plot(losses.index, losses['valid'], label='Valid')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('neural-pix2vid_losses.png')
plt.close()