import numpy as np
from utils_2d.data_utils import *
import os
from utils_2d.utils import *
import functools
from utils_2d.train_sc_utils import *
from config.SC_config_ns import *

### modify dataset
L = 1
### gird
Nx = 256
m = 8
Nx_c = int(Nx/m)

points_x = get_grid(Nx, L)
points_x_0 = get_grid(int(Nx/m), L)
points_x_1 = get_grid(int(Nx/4), L)
points_x_2 = get_grid(int(Nx/2), L)

xx_0, yy_0 = np.meshgrid(points_x_0, points_x_0)
xx_1, yy_1 = np.meshgrid(points_x_1, points_x_1)
xx_2, yy_2 = np.meshgrid(points_x_2, points_x_2)
xx, yy = np.meshgrid(points_x, points_x)
##### number of training ############
N_gen = 4000
N_sup = 4000

N_train = N_gen

N_train_sup = int(N_sup * 0.98)
N_test_sup = int(N_sup * 0.02)

N_test = 100

cwd=os.getcwd()

## two dataset: set 1: high resolution high fidelity dataset
## set 2: uncorrespond low fidelity dataset
Ori_data_name_high = cwd + '/raw_data/NS_high_t_3.npy'
Ori_data_name_test = cwd + '/raw_data/NS_test_t_3.npy'

Gen_data = cwd + '/data/NS_gen_data_N_' + num2str_deciaml(N_gen) + '.npy'
Super_data = cwd + '/data/NS_super_data_N_' + num2str_deciaml(N_sup) + '.npy'
Super_test_data = cwd + '/data/NS_super_test_data_N_' + num2str_deciaml(N_test) + '.npy'
Test_data = cwd + '/data/NS_test_data_N_' + num2str_deciaml(N_test) + '.npy'

######## training strategy for score model #################
batch_size_large = 64
batch_size_small = 20
n_epochs = 4001

save_epoch_large = 500
save_epoch_small = 200

### diffussion model config ###
### define SDE coefficients
sigma_b = 25.0
sigma_s = 25.0
marginal_prob_std_fn_s = functools.partial(marginal_prob_std, sigma=sigma_s)
diffusion_coeff_fn_s = functools.partial(diffusion_coeff, sigma=sigma_s)

marginal_prob_std_fn_b = functools.partial(marginal_prob_std, sigma=sigma_b)
diffusion_coeff_fn_b = functools.partial(diffusion_coeff, sigma=sigma_b)

