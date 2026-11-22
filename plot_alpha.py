#!/usr/bin/env python3.12
# coding: utf-8

import labels as l
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

dat = np.load('alpha.npz')
ff_arr = [0.1, 0.2, 0.4, 0.6, 0.8]
m_arr = [0, 0.5, 0.66, 1]
M_arr = [0.1, 1, 2, 5, 10]
n_sample = np.shape(dat['alpha'])[1]

fig, ax = plt.subplots(figsize=(6,5))
colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = colors[:4]+colors[6:]

M_ind = 0
for i, M in enumerate(M_arr):
    color = colors[i]
    alpha = []
    alpha_err = []
    fraction_used = []
    m_arr_plot = list(m_arr)
    setB_use_prominent = np.zeros(4)
    setB_use_fraction = np.zeros(4)
    for i, m in enumerate(m_arr):
        ind = [element for element in np.where(dat['M']==M)[0] if element in np.where(dat['m']==m)[0]][0]
        alpha_ = dat['alpha'][ind]
        elim_stage = dat['elim_stage'][ind]
        setB_fit_used = dat['setB_fit_used'][ind]

        ind = alpha_!=0
        alpha_ = alpha_[ind]
        setB_fit_used = setB_fit_used[ind]
        if len(alpha_):
            alpha.append(np.percentile(alpha_, 50))
            alpha_err.append([np.percentile(alpha_, 16), np.percentile(alpha_, 84)])
            fraction_used.append(len(alpha_)/n_sample)
            setB_use_fraction[i] = np.mean(setB_fit_used)
            setB_use_prominent[i] = np.mean(setB_fit_used) > 0.1
        else:
            m_arr_plot.remove(m)

    if len(m_arr_plot):
        m_arr_plot = np.array(m_arr_plot) + M_ind*0.02
        alpha_err = np.abs(alpha_err - np.array([alpha, alpha]).T)
        label_used = False
        for i, B in enumerate(setB_use_prominent):
            if B:
                ax.errorbar(m_arr_plot[i], alpha[i] - alpha_err[i][0], yerr=np.array([[1, 0.25]]).T, alpha=0.5, zorder=0, ls='none',fmt='o', markersize=8, capsize=2, color=color, lolims=True)
            else:
                if not label_used:
                    label = r'$%s = %g$'%(l.mach, M)
                    label_used = True
                else:
                    label = None
                ax.errorbar(m_arr_plot[i], alpha[i], yerr=np.array([alpha_err[i]]).T, alpha=0.5, zorder=0, ls='none',fmt='o', markersize=8, capsize=2, color=color, label=label)
        M_ind += 1

plt.axhline(1, c='k', ls='--', label=r'$\alpha=1$', alpha=0.5)
plt.ylim(-0)
plt.xlim(-0.1, 1.2)
plt.xlabel('exponent, m')
plt.ylabel(r'%s'%l.alpha_long)
plt.legend(framealpha=0.4)
plt.savefig('alpha.png')
plt.savefig('alpha.pdf')