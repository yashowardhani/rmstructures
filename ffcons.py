#!/usr/bin/env python3.12
# coding: utf-8

import numpy as np
import cfpack as cfp
import matplotlib.pyplot as plt
import labels as l
import argparse
import inspect
import glob
from utilities import load_dat
from functions import squareroot
from numpy.random import normal

if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create a plot between alpha and Sigma for supplied datfiles')
    parser.add_argument('-m', nargs='*', type=float, default=[0])
    parser.add_argument('-M', nargs='*', type=float, default=[0.1, 1, 2, 5, 10])
    parser.add_argument('-o', '--outname', default='ffcons')
    parser.add_argument('-ov', '--overwrite', default=False, action='store_true')
    args = parser.parse_args()
    plt.rcParams["font.family"] = "serif"

    ff_arr = [0.1, 0.2, 0.6, 0.4, 0.8]
    xlabel = r'$%s$'%l.meanbysigmaDM
    ylabel = r'%s'%l.ff_long
    function = squareroot
    fit_params = inspect.getfullargspec(function).args[1:]
    M_save, psamples_save, rmse_save = [], [], []

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors[:4]+colors[6:]

    plt.figure(figsize=(6,5))

    for i in range(len(args.M)):
        m, M, color = 0, args.M[i], colors[i]
        print('working on M = %g'%M)

        x_var = []
        x_var_err = []
        # loading data data
        fit = load_dat('m%gM%g/stuff.dat'%(m,M), verbose=False)

        # x axis parameter
        x_var = np.array([np.mean(fit.meanbysigmaDM[fit.ff==ff]) for ff in ff_arr])
        x_var_err = np.array([np.std(fit.meanbysigmaDM[fit.ff==ff]) for ff in ff_arr])

        # fit
        fit = cfp.fit(squareroot, x_var, ff_arr, perr_method='systematic')
        opt, err, samples = fit.popt, np.mean(np.abs(fit.perr), axis=1), fit.psamples.T
        x_plot = np.linspace(x_var[0], x_var[-1], 141)

        # median estimates for y; root mean square error
        y_plot_ = []
        ff_fit_ = []
        for opt in samples:
            y_plot = squareroot(x_plot, *opt)
            ff_fit = squareroot(x_var, *opt)
            if np.max(y_plot) == 1e99:    continue
            if np.max(ff_fit) == 1e99:    continue
            y_plot_.append(y_plot)
            ff_fit_.append(ff_fit)
        y_plot = np.median(y_plot_, axis=0)             # fitted values
        y_err0 = np.percentile(y_plot_, 16, axis=0)     # lower limit of error region
        y_err1 = np.percentile(y_plot_, 84, axis=0)     # upper limit of error region
        ff_fit = np.median(ff_fit_, axis=0)
        rms_error = (1/len(x_var) * np.sum((ff_arr - ff_fit)**2))**0.5

        # plotting
        plt.plot(x_plot, y_plot, c='k', ls='--')
        plt.fill_between(x_plot, y_err0, y_err1, alpha=0.1, color=color)
        plt.scatter(x_var, ff_arr, color=color, label='$%s$ = %g'%(l.mach, M))

        # estimating RMSE
        y_fit_ = []
        for opt in fit.psamples.T:
            y_fit = squareroot(x_var, *opt)
            if np.max(y_fit) == 1e99:   continue
            y_fit_.append(y_fit)
        y_fit = np.median(y_fit_, axis=0)

        M_save.append(M)
        psamples_save.append(fit.psamples.T)
        rmse_save.append(rms_error)

    # save psamples
    np.savez('ffcons_psamples.npz', M=M_save, psamples=psamples_save, rms_error=rmse_save)

    # plot formatting
    plt.legend(frameon=True, framealpha=0.95, loc=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    fig = plt.gcf()
    fig.savefig(args.outname + '.pdf')
    fig.savefig(args.outname + '.png')
    print(args.outname, 'saved.')