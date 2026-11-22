#!/usr/bin/env python3.12
# coding: utf-8

from utilities import load_dat
from scipy.stats import spearmanr
from numpy.random import normal, choice
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from functions import squareroot

def function(x,a,b,c):
    return a+c/(b-x)

dat = load_dat('pulsars.dat', verbose=False)
fit = np.load('alpha_psamples.npz')
ffc = np.load('ffcons_psamples.npz')

# for these paths, ff < 0.3 fit data is an "improved" fit
ind = fit['mindisp'] != 0
improved_fit_paths = np.unique(fit['paths'][ind])

ff_arr = [0.1, 0.2, 0.4, 0.6, 0.8]
m_arr = [0, 0.5, 0.66, 1]
M_arr = [0.1, 1, 2, 5, 10]

n_sample = 10000
samples_inds = np.arange(len(fit['psamples'][0]))
ind_dat = np.abs(dat.GB) < 5
fraction_used_ = []

m_save, M_save, ff_save, rho_save, alpha_save, elims_save, setB_fit_used_save = [], [], [], [], [], [], []
M_ind = 0
for M in tqdm(M_arr, desc='M', leave=False):
    ind0 = np.where(ffc['M']==M)[0][0]
    fraction_used = []
    m_arr_plot = list(m_arr)
    for m in tqdm(m_arr, desc='m', leave=False):
        ind1 = np.where(fit['paths'] == 'm%gM%g'%(m, M))[0]
        alpha_, ff_, rho_, setB_fit_used_ = [], [], [], []
        elim_stage = np.zeros(3)
        for j in tqdm(range(n_sample), desc='iter', leave=False):
            DM = normal(dat.DM[ind_dat], dat.DM_err[ind_dat])
            if np.sum(DM<0):
                elim_stage[0] += 1
                continue

            a, b = ffc['psamples'][ind0][choice(samples_inds)]
            ff = squareroot(np.mean(DM)/np.std(DM), a, b)

            # throwing away if negative vol filling fraction
            if ff < 0:
                elim_stage[1] += 1
                continue

            # marking that this iteration uses one of the "improved" fits
            setB_fit_used = 0
            if ff < 0.3:
                if 'm%gM%g'%(m, M) in improved_fit_paths:
                    setB_fit_used = 1

            a_arr, b_arr, c_arr = fit['psamples'][ind1][:,choice(samples_inds)].T
            ff_arr = fit['ff'][ind1]
            f = interp1d(ff_arr, a_arr, fill_value='extrapolate'); a = float(f(ff))
            f = interp1d(ff_arr, b_arr, fill_value='extrapolate'); b = float(f(ff))
            f = interp1d(ff_arr, c_arr, fill_value='extrapolate'); c = float(f(ff))
            RM = normal(dat.RM[ind_dat], dat.RM_err[ind_dat])
            rho = spearmanr(np.abs(RM), DM)[0]

            alpha = function(rho, a, b, c)
            if alpha <= 0:
                elim_stage[2] += 1
                continue

            ff_.append(ff)
            rho_.append(rho)
            alpha_.append(alpha)
            setB_fit_used_.append(setB_fit_used)

        m_save.append(m)
        M_save.append(M)
        fraction_used.append(len(alpha_)/n_sample)
        elims_save.append(elim_stage)

        n_iter = len(ff_)
        ff_save.append(np.pad(ff_, (0, n_sample-n_iter), 'constant'))
        rho_save.append(np.pad(rho_, (0, n_sample-n_iter), 'constant'))
        alpha_save.append(np.pad(alpha_, (0, n_sample-n_iter), 'constant'))
        setB_fit_used_save.append(np.pad(setB_fit_used_, (0, n_sample-n_iter), 'constant'))

np.savez('alpha.npz', m=m_save, M=M_save, ff=ff_save, rho=rho_save, alpha=alpha_save, elim_stage=elims_save, setB_fit_used=setB_fit_used_save)