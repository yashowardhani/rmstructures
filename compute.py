#!/usr/bin/env python
# coding: utf-8

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import h5py
import argparse
from scipy.stats import spearmanr
from tqdm import tqdm
import re

#loads all (or requisite) components of a model
def load_hdf5(filename, datasets=[]):
    with h5py.File(filename, "r") as hf:
        if not len(datasets):
            datasets = list(hf.keys())
        for i in range(len(datasets)):
            globals()[datasets[i]] = np.array(hf[datasets[i]])


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Find certain parameters involving RM and DM data')
    parser.add_argument('-ff', '--fill_fractions', nargs='*', type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("-suff", "--suff", type=str, help="String to append to filename (eg. seed, type, etc)", default="")
    parser.add_argument("-n_bins", "--n_bins", type=int, help="Number of bins used to bin the data in order to fit", default=50)
    parser.add_argument("--magne", action='store_true', default=False)
    parser.add_argument("--ffcons", action='store_true', default=False)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()

    outfile = 'stuff.dat'
    disps = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.5]
    thetas = [0, 30, 60, 90]

    with open(outfile,'w') as f:
        l = (     '{:>10}'.format('ff') 
                + '{:>11}'.format('disp') 
                + '{:>11}'.format('theta') 
                + '{:>25}'.format('alpha') 
                + '{:>25}'.format('rmdmcorr'))
        if args.magne:  l += '{:>25}'.format('magnecorr')
        if args.ffcons: l += '{:>25}'.format('meanbysigmaDM')
        f.write('#'+ l + '\n')
        for ff in args.fill_fractions:
            filenames = []
            for disp in disps:
                for theta in thetas:
                    filenames.append('multiseed/ms_f%gd%gt%d.h5'%(ff, disp, theta))

            for filename in tqdm(filenames):
                load_hdf5(filename)

                #taking only points where DM is nonzero
                ind = DM!=0
                RM = RM[ind]
                mag_av = mag_av[ind]
                DM = DM[ind]

                #RMbyDM vs mag_av
                #generating bin data for plotting and fitting
                pdf_2d, x, y = np.histogram2d(mag_av, RM/DM, bins=args.n_bins, density=False)
                x = (x[:-1] + x[1:])/2
                y = (y[:-1] + y[1:])/2
                x, y = np.meshgrid(x, y, indexing='ij')
                w, x, y = np.ravel(pdf_2d), np.ravel(x), np.ravel(y)

                mask = w > 0

                x = x[mask]
                y = y[mask]
                w = w[mask]

                # directly solving for slope
                m = np.sum(w*x*y) / np.sum(w*x**2)

                split_name = re.split(r'([a-z]+)', filename[11:-3])[1:]

                l = ( '{:>10}'.format(split_name[split_name.index('f')+1])
                    + '{:>11}'.format(split_name[split_name.index('d')+1])
                    + '{:>11}'.format(split_name[split_name.index('t')+1])
                    + '{:>25}'.format(m)
                    + '{:>25}'.format(spearmanr(np.abs(RM), DM)[0]))
                if args.magne:  l += '{:>25}'.format(spearmanr(np.abs(mag), ne)[0])
                if args.ffcons: l += '{:>25}'.format(np.mean(DM)/np.std(DM))

                f.write(l + '\n')