#!/usr/bin/env python3.12
# coding: utf-8

import labels as l
import matplotlib as mpl
from utilities import load_hdf5
import numpy as np
import matplotlib.pyplot as plt

nbins = 60
norm = mpl.colors.LogNorm(vmin=1, vmax=2e3)
disps = [0, 0.5]
thetas = [0, 90]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,13))
plt.subplots_adjust(wspace=0, hspace=0.1)
images = []
xmax, ymax = 0.7, 4
xedges = np.linspace(-xmax, xmax, nbins)
yedges = np.linspace(-ymax, ymax, nbins)

for i, disp in enumerate(disps):
    for j, theta in enumerate(thetas):
        myax = axes[i][j]
        out = load_hdf5('m0.5M2/multiseed/ms_f0.1d%gt%d.h5'%(disp, theta))

        # data, bin edges
        ind = out.DM!=0
        x = out.mag_av[ind]
        y = out.RM[ind]/out.DM[ind]

        # histogramming, plotting
        H, xedges, yedges, image = myax.hist2d(x, y, bins=(xedges, yedges), norm=norm)

        # flattening histogrammed data for fitting
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
        mask = H > 0
        x = X[mask]
        y = Y[mask]
        w = H[mask]

        # directly solving for slope and plotting fit line
        m = np.sum(w*x*y) / np.sum(w*x**2)
        x_arr = np.linspace(-xmax, xmax, 100)
        myax.plot(x_arr, m*x_arr, 'k--', label='slope = %s'%round(m,2))

        # x = y line
        myax.plot(x_arr, x_arr, c='m', alpha=0.5, lw=3, label='$x=y$ line')

        # presentation 
        myax.set_box_aspect(1)
        myax.legend(loc=4)
        myax.text(-0.55, 3, r'$%s, %s = %g, %d^\circ$'%(l.disp, l.theta, disp, theta))
        if i%2: myax.set_xlabel(r'$%s$'%l.bav, labelpad=10)
        else:   myax.get_xaxis().set_ticklabels([])
        if j%2: myax.get_yaxis().set_ticklabels([])
        else:   myax.set_ylabel(r'$%s$'%l.RMbyDM, labelpad=-0.1)

plt.colorbar(image, ax=axes, norm=norm, aspect=30, orientation='horizontal', fraction=0.04, pad=0.08)
plt.savefig('histograms.pdf')
plt.savefig('histograms.png')