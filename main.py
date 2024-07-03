import pandas as pd
import torch.optim as optim
from neuralop.models import *
from torch.utils.data import DataLoader, random_split

from utils import *

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

device = check_torch()
    
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
        print('Epoch: {} | Loss: {:.4f} | Valid Loss: {:.4f}'.format(epoch, train_loss[-1], valid_loss[-1]))

torch.save(model.state_dict(), 'MiONet.pth')
losses = pd.DataFrame({'train': train_loss, 'valid': valid_loss})
losses.to_csv('MiONet_losses.csv', index=False)