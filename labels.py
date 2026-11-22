DM = r'\mathrm{DM}'
RM = r'\mathrm{RM}'
sigmaDM = r'\sigma_{%s}'%DM
meanDM = r'\mu_{%s}'%DM
meanbysigmaDM = r'%s/%s'%(meanDM, sigmaDM)
meanRMabs = r'\mu_{|%s|}'%RM
sigmaRM = r'\sigma_{%s}'%RM
absRM = r'\vert%s\vert'%RM
ff = r'f_\mathrm{vol}'
bpar = r'B_\parallel'
absbpar = r'\vert %s \vert'%bpar
Ne = r'n_\mathrm{e}'
bav = r'\langle {%s} \rangle'%bpar
bDMabsRM = r'\zeta'
alpha = r'\alpha'
alphanaught = r'\alpha_0'
krm = r'\kappa'
RMbykrm = r'%s/%s'%(RM, krm)
RMbyDM = r'%s/%s%s'%(RM, krm, DM)
sigmaRMbyDM = r'\sigma_{\tiny{%s}}'%RMbyDM
normsigma = r'\Sigma'
normsigmanaught = r'\Sigma_0'
coefficient = r'\gamma'
disp = r'd/L'
theta = r'\theta'
dtheta = r'%s, \theta'%(disp)
ffcons = r'%s/%s'%(meanDM,sigmaDM)
meanNe = r'\mu_{%s}'%(Ne)
sigmaB = r'\sigma_{%s}'%(bpar)
mach = r'\mathcal{M}'
n = r'\textrm{N}'
magnecorr = r'\rho_\textrm{3D}'
rmdmcorr = r'\rho'
rmdmcorrnaught = r'\rho_0'
exponent = r'\mathrm{m}'


bDMabsRM_long = r'$%s$ (intercept of $%s$ vs $%s$)'%(bDMabsRM, DM, absRM)
alpha_long = r'$%s$ (slope between $%s$ and $%s$)'%(alpha, RMbyDM, bav)
sigmaRMbyDM_long = r'$%s$ (standard deviation of $%s$)'%(sigmaRMbyDM, RMbyDM)
magnecorr_long = r'$%s$ (correlation between $%s$ and $%s$)'%(magnecorr, Ne, absbpar)
rmdmcorr_long = r'$%s$ (correlation between $%s$ and $%s$)'%(rmdmcorr, absRM, DM)
ff_long = r'$%s$ (volume filling fraction)'%ff
normsigma_long = r'$%s$ = $%s \times %s / %s$'%(normsigma, sigmaRMbyDM, meanDM, sigmaRM)
disp_long = r'displacement, $%s$'%disp
theta_long = r'rotation, $\theta$'

