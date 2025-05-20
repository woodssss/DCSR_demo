import os
import functools
from utils_1d.train_sc_utils import *

Nx = 256
l = 0.3
Lx = 1
points = np.linspace(0, Lx, Nx)
Ns = 2000
v, dt, T = 0.1, 0.001, 0.25

cwd=os.getcwd()

Data_name = cwd + '/data/Bumps_data.npy' # this should include ic, solu, pred from lw, godn, cd


### define SDE coefficients
sigma_s = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma_s)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma_s)

