#!/usr/bin/env python
# coding: utf-8

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import h5py
import argparse
from tqdm import tqdm

#loads all (or requisite) components of a model
def load_hdf5(filename, datasets=[]):
    with h5py.File(filename, "r") as hf:
        if not len(datasets):
            datasets = list(hf.keys())
        for i in range(len(datasets)):
            globals()[datasets[i]] = np.array(hf[datasets[i]])


# ===== the following applies in case we are running this in script mode =====
# ========================== MAIN ============================================
# ## Loop Tests for Theta and STD
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Collect all 2D data from models of different seeds for further processing')
    parser.add_argument("-i", "--i", dest='filenames', nargs='*', help="Input h5 files", required=False, default=None)
    parser.add_argument('-ff', '--fill_fractions', nargs='*', type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("-s", "--seeds", nargs='*', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--save_magne", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    
    args = parser.parse_args()

    disps = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.5]
    thetas = [0, 30, 60, 90]

   
    if args.filenames is None:
        filenames = []
        for ff in args.fill_fractions:
            for disp in disps:
                for theta in thetas:
                    filenames.append('ellipsoids_f%gd%gt%d.h5'%(ff, disp, theta))
    else:
        filenames = args.filenames

    d = 'multiseed'
    os.makedirs(d, exist_ok=True)
    load_hdf5('s1/%s'%filenames[0], ['mag'])
    n = np.shape(mag)[0]
        
    for filename in tqdm(filenames):
        dest = d+'/'+filename.replace('ellipsoids','ms')
        #aggregating data from different seeds....
        RM, DM, mag_av = [], [], []
        if args.save_magne: mag_all, ne_all = [], []
        for seed in args.seeds:
            load_hdf5('s%s/%s'%(seed, filename), ['mag', 'ne'])
            for axis in [0,1,2]:
                mag_av.extend(list(np.ravel(np.sum(mag *(1.0/n), axis=axis))))
                DM.extend(list(np.ravel(np.sum(ne *(1.0/n), axis=axis))))
                RM.extend(list(np.ravel(np.sum(mag*ne *(1.0/n), axis=axis))))
            if args.save_magne:
                mag_all.extend(list(np.ravel(mag)))
                ne_all.extend(list(np.ravel(ne)))

        with h5py.File(dest, 'w') as hf:
            if args.save_magne:
                hf.create_dataset('mag', data=mag_all)
                hf.create_dataset('ne', data=ne_all)
            hf.create_dataset('RM', data=RM)
            hf.create_dataset('DM', data=DM)
            hf.create_dataset('mag_av', data=mag_av)
    print('')