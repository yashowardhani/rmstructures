#!/usr/bin/env python3.12
# coding: utf-8

import os
import numpy as np
from cfpack import fit as cfp_fit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import labels as l
import argparse
import inspect
import glob
from utilities import load_dat, darken
import matplotlib as mpl

def fit_function(x,a,b,c):
    x = np.array(x)
    ret = np.where(x>b, 1e99, a+c/(b-x))
    if np.any(np.isnan(ret)):   ret[np.isnan(ret)] = 1e99
    return ret

def main_plot(x_var, y_var, disp, theta, extreme_points=False, calc_fit=False, params=None, perr_method='systematic', param_names=None, fit_method='cfp', separate_by_disp=False):
    sort = x_var.argsort()
    x_var = x_var[sort]
    y_var = y_var[sort]
    disp = disp[sort]
    theta = theta[sort]
    fit_params = inspect.getfullargspec(fit_function).args[1:]

    if param_names == None:
        param_names = fit_params

    # composing a label and other things necessary for plotting
    if len(args.paths) == 1:    label_scatter = r'$%s$ = %g'%(l.ff, ff)
    else:                       label_scatter = r'%s, $%s$ = %g'%(path, l.ff, ff)

    if calc_fit:
        x_plot = np.linspace(np.min(x_var)-0.0015, np.max(x_var), 200)
        match fit_method:
            case 'cfp':
                params = {
                    'a':    [-10, 0, 1],
                    'b':    [np.max(x_var), 1, 10],
                    'c':    [1e-6, 1, 10]
                }
                fit = cfp_fit(fit_function, x_var, y_var, xerr=None, yerr=None, verbose=0, params=params, perr_method=perr_method)
                popt, psamples = fit.popt, fit.psamples.T

                y_plot_ = []
                popt_ = psamples
                for opt in popt_:
                    y_plot = fit_function(x_plot, *opt)
                    if np.max(y_plot) == 1e99:    continue
                    y_plot_.append(y_plot)
                y_err0 = np.percentile(y_plot_, 16, axis=0)
                y_err1 = np.percentile(y_plot_, 84, axis=0)
                plt.fill_between(x_plot, y_err0, y_err1, alpha=0.2, color=color)
                y_plot = np.median(y_plot_, axis=0)

            case 'curve_fit':
                fit = curve_fit(fit_function, x_var, y_var)
                popt, pcov = fit
                y_plot = fit_function(x_plot, *popt)

        plt.plot(x_plot, y_plot, lw=2, zorder=5, c=darken(color, 0.1), linestyle=(0, (5, 1)))


    # plotting extreme data points
    if extreme_points:
        for i in range(len(disp)):
            match disp[i], theta[i]:
                case 0.0,  0:    marker="^"; label=r'$d, \theta$ = 0,0'
                case 0.0, 90:    marker="<"; label=r'$d, \theta$ = 0,90$^{\circ}$'
                case 0.5,  0:    marker=">"; label=r'$d, \theta$ = 0.5,0'
                case 0.5, 90:    marker="v"; label=r'$d, \theta$ = 0.5,90$^{\circ}$'
                case   _,  _:    continue       # do not plot any non-extreme points
            if not recur_label: label = None    # to avoid recurrant labels for the same thing
            plt.scatter(x_var[i], y_var[i], marker=marker,c='white',edgecolors='k',label=label, zorder=10)

    # plotting the real data scatter
    if not separate_by_disp:
        plt.scatter(x_var, y_var, alpha=0.5, label=label_scatter, c=color, zorder=0)
    else:
        colors = mpl.colormaps['turbo'](np.linspace(0, 1, len(d_range)))
        for i,d in enumerate(d_range):
            #if d<0.03:  color=colors[1]
            #else:       color=colors[2]
            #label=None
            #if d==0:    label=r"$d \lt 0.03$"
            #if d==0.03: label=r"$d \geq 0.03$"
            ind = np.where(disp==d)[0]
            #plt.scatter(x_var[ind], y_var[ind], alpha=0.7, label=label, zorder=0, c=color)
            plt.scatter(x_var[ind], y_var[ind], alpha=0.7, label=r'$d = %g$'%d, zorder=0, color=colors[i])

    if calc_fit:    return psamples


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Create a plot between alpha and Sigma for supplied datfiles')
    parser.add_argument("-p", "--paths", nargs="*", help="Paths for the necessary datfiles", default=['m0.5M2'])
    parser.add_argument('-ff', '--fill_fractions', nargs='*', type=float, help="Fill fraction", default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--fit", type=int, help="which fit?", default=1)
    parser.add_argument('-nf', '--nofit', action='store_true', default=False)
    parser.add_argument("--overwrite", action='store_true', default=True)
    parser.add_argument('-s', '--suffix', default='')
    parser.add_argument('-o', '--outname', default=None)
    args = parser.parse_args()
    plt.rcParams["font.family"] = "serif"

    recur_label = True

    if args.outname is not None:    plotname = args.outname
    else:                           plotname = 'main_fit%s'%args.suffix
    
    #if args.fill_fractions is None:
    #    match len(args.paths):
    #        case 1: args.fill_fractions = [0.1, 0.2, 0.4, 0.6, 0.8]
            #case _: args.fill_fractions = [0.2]

    param_names = [l.alphanaught, l.rmdmcorrnaught, l.coefficient]

    selective_dtrange = True
    d_range = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.5]
    t_range = [0, 30, 60, 90]

    if plotname+'.pdf' in glob.glob('*') and not args.overwrite:
        print('main plot already present. skipping...')
    else:
        yname = 'alpha'
        xname = 'rmdmcorr'
        calc_fit = 0
        save_psamples = 0

        xlabel = r'%s'%getattr(l, xname+'_long')
        ylabel = r'%s'%getattr(l, yname+'_long')


        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = colors[:4]+colors[6:]

        # for every disp different colour
        #n_lines = len(d_range)
        #colors = mpl.colormaps['spring'](np.linspace(0, 0.7, n_lines))

        if calc_fit and save_psamples:  path_save, ff_save, psamples_save = [], [], []
        plots_dir = '%s_fits'%yname

        plt.figure(figsize=(6,5))
        for path_index, path in enumerate(args.paths):
            data = load_dat(os.path.join(path, 'stuff.dat'))
            for ff_index, ff in enumerate(args.fill_fractions):
                print('working on %s, %s'%(path, ff))
                color = colors[path_index + ff_index]
                #color = colors[ff_index]
                # fetching indices for given ff
                match_index = np.where(data.ff == ff)[0]
                if selective_dtrange:
                    match_index = [i for i in match_index if data.disp[i] in d_range and data.theta[i] in t_range]

                x = getattr(data, xname)[match_index]
                y = getattr(data, yname)[match_index]

                disp = data.disp[match_index]
                theta = data.theta[match_index]

                # plot statement
                psamples = main_plot(x, y, disp, theta, 
                    extreme_points = 0,
                    calc_fit = calc_fit,
                    #fit_method = 'curve_fit',
                    #separate_by_disp = True,
                    )
                recur_label = False

                if calc_fit and save_psamples:
                    path_save.append(path)
                    ff_save.append(ff)
                    psamples_save.append(psamples)

        # plot formatting
        plt.legend(frameon=True, framealpha=0.95)
        if [yname, xname] == ['alpha', 'magnecorr']:
            plt.ylim(0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        fig = plt.gcf()
        #plotname = '%s/%s'%(plots_dir, path)
        #fig.savefig(plotname+'.pdf')
        fig.savefig('pvar_plots/' + plotname+'.pdf')
        fig.savefig('pvar_plots/' + plotname+'.png')
        print('%s saved.'%plotname)

        if calc_fit and save_psamples:
            np.savez('%s_psamples.npz'%yname, paths=path_save, ff=ff_save, psamples=psamples_save)