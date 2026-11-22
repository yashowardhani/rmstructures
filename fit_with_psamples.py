#!/usr/bin/env python3.12
# coding: utf-8

import os
import numpy as np
from cfpack import fit
import argparse
import glob
from tqdm import tqdm
from utilities import load_dat

def fit_function(x,a,b,c):
    x = np.array(x)
    ret = np.where(x>b, 1e99, a+c/(b-x))
    index = np.isnan(ret)
    if np.any(index):
        ret[index] = 1e99
    return ret

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create a plot between alpha and Sigma for supplied datfiles')
    parser.add_argument("-p", "--paths", nargs="*", help="Paths for the necessary datfiles", required=False, default=[])
    parser.add_argument('-ff', '--fill_fractions', nargs='*', type=float, help="Fill fraction", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    args = parser.parse_args()

    if len(args.paths)==0: args.paths = glob.glob("m*M*")

    path_save, ff_save, psamples_save, old_nrmse_save, nrmse_save, mindisp_save = [], [], [], [], [], []

    yname = 'alpha'
    xname = 'rmdmcorr'

    selective_dtrange = True
    d_range = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.5]
    t_range = [0, 30, 60, 90]

    for path in tqdm(args.paths):
        m, mach = map(float, path[1:].split('M'))
        data = load_dat(os.path.join(path, 'stuff.dat'), verbose=False)

        for ff_index, ff in enumerate(args.fill_fractions):

            # x and y axis parameters
            match_index = np.where(data.ff == ff)[0]
            if selective_dtrange:
                match_index = [i for i in match_index if data.disp[i] in d_range and data.theta[i] in t_range]

            x = list(getattr(data, xname)[match_index])
            y = list(getattr(data, yname)[match_index])
            disp = list(getattr(data, 'disp')[match_index])

            first_iteration = True
            old_nrmse = None

            while 1:
                # fitting
                params = {
                    'a':    [-10, 0, 1],
                    'b':    [np.max(x), 1, 10],
                    'c':    [1e-6, 1, 10]
                }

                f = fit(fit_function, x, y, verbose=0, params=params, perr_method='systematic')
                popt, psamples = f.popt, f.psamples.T

                # calculating NRMSE
                y_fit_ = []
                for opt in psamples:
                    y_fit = fit_function(x, *opt)
                    if np.max(y_fit) == 1e99:   continue
                    y_fit_.append(y_fit)
                y_fit = np.median(y_fit_, axis=0)
                rmse = (1/len(x) * np.sum((y - y_fit)**2))**0.5
                nrmse = rmse/np.median(y)

                if nrmse > 0.2:
                    if first_iteration:
                        old_nrmse = nrmse
                        first_iteration = False
                    rem_indices = np.where(disp == np.min(disp))[0]
                    for i in sorted(rem_indices, reverse=True):
                        del disp[i]
                        del x[i]
                        del y[i]
                else:
                    break

            path_save.append(path)
            ff_save.append(ff)
            psamples_save.append(psamples)
            old_nrmse_save.append(old_nrmse)
            nrmse_save.append(nrmse)
            mindisp_save.append(np.min(disp))

    np.savez('%s_psamples.npz'%yname, paths=path_save, ff=ff_save, psamples=psamples_save, nrmse=nrmse_save, old_nrmse=old_nrmse_save, mindisp=mindisp_save)