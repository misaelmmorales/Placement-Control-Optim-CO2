import numpy as np
import matplotlib.pyplot as plt

def plot_control(wlist, wctrl, timesteps, nw:int=5, nt:int=33, figsize=(15,6), cmap:str='inferno',
                 vmin=0, vmax=30):
    plt.figure(figsize=figsize)

    plt.subplot(121)
    im = plt.imshow(wctrl, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(nw), ['W{}'.format(i+1) for i in range(nw)])
    plt.yticks(np.arange(nt), np.arange(1,nt+1))
    plt.ylabel('Timestep')
    plt.colorbar(im, pad=0.04, fraction=0.046)

    plt.subplot(122)
    plt.hlines(0, 0, 10, color='k', alpha=0.5)
    for w in range(wlist.shape[-1]):
        for t in range(1,28):
            plt.hlines(wctrl[t,w], timesteps[t-1], timesteps[t], color='C{}'.format(w))
            plt.vlines(timesteps[t], wctrl[t,w], wctrl[t+1,w], color='k', ls=':')
    plt.xticks(np.arange(10.5, step=0.5))
    plt.xlabel('Time [years]')
    plt.ylabel('MT CO2 injected')
    plt.xlim(0, 10)
    plt.ylim(-0.1, vmax)
    plt.grid(True, which='both', alpha=0.4)

    plt.tight_layout()
    plt.show()
    return None