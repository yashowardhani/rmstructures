#!/usr/bin/env python3.12
# coding: utf-8

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import labels as l
from utilities import load_hdf5
import numpy as np
import matplotlib.pyplot as plt

out = load_hdf5('m0.5M2/s1/ellipsoids_f0.2d0.5t90.h5')

n = 200
DM = np.sum(out.ne *(1.0/n), axis=0)
RM = np.sum(out.mag * out.ne *(1.0/n), axis=0)
mag_av = np.sum(out.mag *(1.0/n), axis=0)
RMbyDM = np.where(DM!=0, RM/DM, 0)

# RM and DM maps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,11.5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
label_key = ['RMbykrm', 'DM']
cmaps = ['seismic', 'gist_heat_r']
lims = [[-0.2, 0.2], [None, 1]]

for i, dset in enumerate(['RM', 'DM']):
    mapdata = globals()[dset]
    myax = ax[i]

    im = myax.imshow(mapdata, vmin=lims[i][0], vmax=lims[i][1], cmap=cmaps[i])
    divider = make_axes_locatable(myax)
    cax = divider.append_axes('bottom', size='5%', pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label(r"$%s$"%getattr(l, label_key[i]))
    myax.xaxis.tick_top()
plt.savefig('maps0.png')
plt.savefig('maps0.pdf')

# bav and RM/DM maps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,11.5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
label_key = ['RMbyDM', 'bav']
lims = [0.6, 0.1, 0.8, 0.8]

gradient = LinearSegmentedColormap.from_list('Random gradient 3470', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%203470=13A900-00FF86-FFFFFF-00C2FF-1269FF
    (0.000, (0.075, 0.663, 0.000)),
    (0.250, (0.000, 1.000, 0.525)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.750, (0.000, 0.761, 1.000)),
    (1.000, (0.071, 0.412, 1.000))))

for i, dset in enumerate(['RMbyDM', 'mag_av']):
    #matdata = getattr(out, dset)
    mapdata = globals()[dset]
    myax = ax[i]

    im1 = myax.imshow(mapdata, vmin=-0.8, vmax=0.8, cmap=gradient)
    divider = make_axes_locatable(myax)
    cax = divider.append_axes('bottom', size='5%', pad=0.1)
    cb = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cb.set_label(r"$%s$"%getattr(l, label_key[i]))
    myax.xaxis.tick_top()
plt.savefig('maps1.png')
plt.savefig('maps1.pdf')