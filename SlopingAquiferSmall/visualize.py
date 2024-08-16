import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils import Units, check_torch, pix2vid_dataset, NeuralPix2Vid

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

tempxt = Xt[:20]
tempct = ct[:20]
print('Xt: {} | ct: {}'.format(tempxt.shape, tempct.shape))
tempy1t = y1t[:20]
tempy2t = y2t[:20]
print('y1t: {} | y2t: {}'.format(tempy1t.shape, tempy2t.shape))

tempx = Xt[:20].detach().cpu().numpy()
tempc = ct[:20].detach().cpu().numpy()
print('Xt: {} | ct: {}'.format(tempx.shape, tempc.shape))
tempy1 = y1t[:20].detach().cpu().numpy()
tempy2 = y2t[:20].detach().cpu().numpy()
print('y1t: {} | y2t: {}'.format(tempy1.shape, tempy2.shape))

tempy1p, temp2yp = model(tempxt, tempct)
tempy1p, tempy2p = tempy1p.detach().cpu().numpy(), temp2yp.detach().cpu().numpy()
print('y1p: {} | y2p: {}'.format(tempy1p.shape, temp2yp.shape))

fig = plt.figure(figsize=(15,10))
gs = GridSpec(2, 1, figure=fig)
sub1 = fig.add_subfigure(gs[0])
sub2 = fig.add_subfigure(gs[1])
labels = ['Poro','LogPerm','Tops','Wells']
gs1 = GridSpec(4, 10, figure=sub1)
for i in range(4):
    for j in range(10):
        ax = sub1.add_subplot(gs1[i,j])
        im = ax.imshow(tempx[j,i], cmap='jet')
        w = np.argwhere(tempx[j,i])
        ax.scatter(w[:,1], w[:,0], c='w', s=5) if i==3 else None
        ax.set(title='R{}'.format(j)) if i==0 else None
        ax.set(ylabel=labels[i]) if j==0 else None
        ax.set(xticks=[], yticks=[])
gs2 = GridSpec(1, 10, figure=sub2)
for j in range(10):
    ax = sub2.add_subplot(gs2[j])
    c = tempc[j]
    im = ax.imshow(np.ma.masked_where(c==0, c), cmap='jet')
    ax.set(title='R{}'.format(j))
    nc = len(np.nonzero(np.sum(tempc[j],0))[0])
    ax.set_xticks(range(nc), labels=np.arange(1,nc+1))
    ax.set_yticks(range(NT//2), labels=np.arange(0.5,10.5,0.5))
    ax.set(ylabel='Timesteps [yr]') if j==0 else None
plt.tight_layout()
plt.savefig('figures/inputs.png', dpi=600)
plt.show()

k = 0
hues = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
fig, axs = plt.subplots(4, 5, figsize=(15,8), sharex=True, sharey=False)
for i in range(4):
    for j in range(5):
        ax = axs[i,j]
        cc = tempc[k]
        nc = len(np.nonzero(np.sum(cc,0))[0])
        for c in range(nc):
            for t in range(1,NT//2+1):
                ax.hlines(cc[t-1,c], t-1, t, color=hues[c])
            for t in range(NT//2):
                ax.vlines(t, cc[t-1,c], cc[t,c], color=hues[c], ls=':', lw=0.75)
        ax.set(title='R{} | ({})'.format(k, nc))
        ax.set(xlim=(0,NT//2))
        ax.grid(True, which='both', alpha=0.25)
        k += 1
fig.text(0, 0.5, 'Injection Rate [MT CO$_2$]', fontsize=14, ha='center', va='center', rotation='vertical')
fig.text(0.5, 0, 'Time [yr]', fontsize=14, ha='center', va='center')
plt.tight_layout()
plt.savefig('figures/controls.png', dpi=600)
plt.show()

fig, axs = plt.subplots(5, 10, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax = axs[i,j]
        k = j * 2 + 1
        ax.imshow(tempy1[i,k,0], cmap='turbo')
        ax.set(title='t={:.1f}'.format(timesteps[k])) if i==0 else None
        ax.set(ylabel='R{}'.format(i)) if j==0 else None
plt.tight_layout()
plt.savefig('figures/y1_true_p.png', dpi=600)
plt.show()

fig, axs = plt.subplots(5, 10, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax = axs[i,j]
        k = j * 2 + 1
        ax.imshow(tempy1[i,k,-1], cmap='turbo', vmin=0, vmax=1)
        ax.set(title='t={:.1f}'.format(timesteps[k])) if i==0 else None
        ax.set(ylabel='R{}'.format(i)) if j==0 else None
plt.tight_layout()
plt.savefig('figures/y1_true_s.png', dpi=600)
plt.show()

fig, axs = plt.subplots(5, 10, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax = axs[i,j]
        k = j * 2 + 1
        ax.imshow(tempy2[i,k,-1], cmap='turbo', vmin=0, vmax=1)
        ax.set(title='t={:.1f}'.format(timesteps[20+k])) if i==0 else None
        ax.set(ylabel='R{}'.format(i)) if j==0 else None
plt.tight_layout()
plt.savefig('figures/y2_true_s.png', dpi=600)
plt.show()

fig, axs = plt.subplots(5, 10, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax = axs[i,j]
        k = j * 2 + 1
        ax.imshow(tempy1p[i,k,0], cmap='turbo')
        ax.set(title='t={:.1f}'.format(timesteps[k])) if i==0 else None
        ax.set(ylabel='R{}'.format(i)) if j==0 else None
plt.tight_layout()
plt.savefig('figures/y1_pred_p.png', dpi=600)
plt.show()

fig, axs = plt.subplots(5, 10, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax = axs[i,j]
        k = j * 2 + 1
        ax.imshow(tempy1p[i,k,-1], cmap='turbo', vmin=0, vmax=1)
        ax.set(title='t={:.1f}'.format(timesteps[k])) if i==0 else None
        ax.set(ylabel='R{}'.format(i)) if j==0 else None
plt.tight_layout()
plt.savefig('figures/y1_pred_s.png', dpi=600)
plt.show()

fig, axs = plt.subplots(5, 10, figsize=(15,6), sharex=True, sharey=True)
for i in range(5):
    for j in range(10):
        ax = axs[i,j]
        k = j * 2 + 1
        ax.imshow(tempy2p[i,k,-1], cmap='turbo', vmin=0, vmax=1)
        ax.set(title='t={:.1f}'.format(timesteps[20+k])) if i==0 else None
        ax.set(ylabel='R{}'.format(i)) if j==0 else None
plt.tight_layout()
plt.savefig('figures/y2_pred_s.png', dpi=600)
plt.show()