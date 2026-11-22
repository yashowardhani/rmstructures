#!/usr/bin/env python3.12
# coding: utf-8

import matplotlib as mpl
import labels as l
from utilities import load_hdf5
import numpy as np
import matplotlib.pyplot as plt

n=200
out = load_hdf5('m0.5M2/s1/ellipsoids_f0.1d0.5t90.h5')

X,Y,Z = np.mgrid[0:n, 0:n, 0:n]
x,y,z = X.flatten(), Y.flatten(), Z.flatten()
val = out.mag.flatten()

x = x[val != 0]
y = y[val != 0]
z = z[val != 0]
val = val[val != 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter3D(x, y, z, c=val, cmap='seismic', norm=mpl.colors.Normalize(-3,3))
cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.1, aspect=15, orientation='horizontal', label=r"$%s$"%l.bav)

plt.minorticks_off()
cb.ax.minorticks_off()
plt.savefig('mag_3d.pdf')
plt.savefig('mag_3d.png')
print('saved mag.')

X,Y,Z = np.mgrid[0:n, 0:n, 0:n]
x,y,z = X.flatten(), Y.flatten(), Z.flatten()
val = out.ne.flatten()

x = x[val != 0]
y = y[val != 0]
z = z[val != 0]
val = val[val != 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter3D(x, y, z, c=val, cmap='gist_heat_r', norm=mpl.colors.Normalize(0, 3))
cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.1, aspect=15, orientation='horizontal', label=r"$%s$"%l.Ne)

plt.minorticks_off()
cb.ax.minorticks_off()
plt.savefig('ne_3d.pdf')
plt.savefig('ne_3d.png')
print('saved ne.')