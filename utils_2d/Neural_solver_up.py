import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils_2d.data_utils import *

def transfer_SOT(T, X, z_size):
    X = X if torch.is_tensor(X) else torch.from_numpy(X).float().to(device)
    X = X.permute(0, 3, 1, 2)
    Nx_c = X.shape[-1]
    Z = torch.randn(X.shape[0], z_size, 1, Nx_c, Nx_c, dtype=torch.float).to(device)
    # print(X.shape, Z.shape)
    # zxc
    with torch.no_grad():
        XZ = torch.cat([X[:,None].repeat(1,z_size,1,1,1), Z], dim=2).to(device)

    with torch.no_grad():
        pd = T(XZ.flatten(start_dim=0, end_dim=1)).permute(1, 2, 3, 0).reshape(1, Nx_c, Nx_c, XZ.shape[0], z_size).permute(3, 4, 0, 1, 2).to('cpu')

    return (pd[:, 0, ...].permute(0, 2, 3, 1)).detach().cpu().numpy()

def transfer_DDIB(marginal_prob_std_fn, diffusion_coeff_fn, model_coarse_gen, model_down_gen, X, eps, t1, t2):
    #ic = torch.from_numpy(X).float().to(device)
    ic = X if torch.is_tensor(X) else torch.from_numpy(X).float().to(device)

    latent = ode_solver(model_coarse_gen, marginal_prob_std_fn, diffusion_coeff_fn, ic, forward=1,
                        eps=eps, T=t1)

    sample_down = ode_solver(model_down_gen, marginal_prob_std_fn, diffusion_coeff_fn, latent, forward=2, eps=eps, T=t2)

    return sample_down.detach().cpu().numpy()


def transfer_sdit(marginal_prob_std_fn, diffusion_coeff_fn, model_down_gen, X, eps, t1, t2):
    ic = X if torch.is_tensor(X) else torch.from_numpy(X).float().to(device)

    latent = get_perturbed_x(ic, marginal_prob_std_fn, t=t1)

    sample_down = ode_solver(model_down_gen, marginal_prob_std_fn, diffusion_coeff_fn, latent, forward=2, eps=eps, T=t2)

    return sample_down.detach().cpu().numpy()

def Sup_up_one(marginal_prob_std_fn, diffusion_coeff_fn, point_x_c, point_x, L, Nx, model_up, X, eps, t, method='ode_solver_cond'):
    ### X [bs, 16, 16, 1]
    ### first interp to [bs, 32, 32, 1] then normal
    bs = X.shape[0]
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
    X_interp_np = interp_pbc_2d_batch(point_x, point_x_c, L, X_np)
    X_interp = torch.from_numpy(X_interp_np).float().to(device)
    X_sample = gen_sample_cond(model_up, X_interp, Nx, bs, 1, marginal_prob_std_fn, diffusion_coeff_fn, method=method, eps=eps, T=t)
    return X_sample

####################### 256x256 #####################################################################
def Sup_up_256(marginal_prob_std_fn, diffusion_coeff_fn, point_x_0, point_x_1, point_x_2, point_x, L, model_up_0, model_up_1, model_up_2, X, eps, t):
    X_64 = Sup_up_one(marginal_prob_std_fn, diffusion_coeff_fn, point_x_0, point_x_1, L, 64, model_up_0, X, eps, t)
    X_128 = Sup_up_one(marginal_prob_std_fn, diffusion_coeff_fn, point_x_1, point_x_2, L, 128, model_up_1, X_64, eps, t)
    X_256 = Sup_up_one(marginal_prob_std_fn, diffusion_coeff_fn, point_x_2, point_x, L, 256, model_up_2, X_128, eps, t)
    return X_64.detach().cpu().numpy(), X_128.detach().cpu().numpy(), X_256.detach().cpu().numpy()

def direct_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_0, point_x_1, point_x_2, point_x, L, eps, t, X, model_up_0, model_up_1=None, model_up_2=None, method1='ode_solver_cond', method2='ode_solver_cond', method3='ode_solver_cond', eps1=1e-5, eps2=1e-5, eps3=1e-5):
    ### given ic, do adapt + fno evo + sup
    out_ls = []
    X_64 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_0, point_x_1, L, 64, model_up_0, X, eps1, t, method=method1)
    X_64 = X_64.detach().cpu().numpy()
    tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
    X_64 /= tmp_nor
    out_ls.append(X_64)

    X_64 = torch.from_numpy(X_64).float().to(device)

    if model_up_1:
        print('Sup 64x64->128x128')
        X_128 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_1, point_x_2, L, 128, model_up_1, X_64,
                           eps2, t, method=method2)
        X_128 = X_128.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
        X_128 /= tmp_nor
        out_ls.append(X_128)
        X_128 = torch.from_numpy(X_128).float().to(device)

        if model_up_2:
            print('Sup 128x128->256x256')
            X_256 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_2, point_x, L, 256, model_up_2,
                               X_128, eps3, t, method=method3)
            X_256 = X_256.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
            X_256 /= tmp_nor
            out_ls.append(X_256)

    return out_ls

def transfer_sdit_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, marginal_prob_std_fn_b, diffusion_coeff_fn_b, point_x_0, point_x_1, point_x_2, point_x, L, eps, t1, t2, t, X, model_down_gen, model_up_0=None, model_up_1=None, model_up_2=None, method1='ode_solver_cond', method2='ode_solver_cond', method3='ode_solver_cond', eps1=1e-5, eps2=1e-5, eps3=1e-5):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_sdit(marginal_prob_std_fn_b, diffusion_coeff_fn_b, model_down_gen, X, eps, t1, t2)

    out_ls.append(Y)

    if model_up_0:
        norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
        Y_nor = Y / norf
        Y_nor = torch.from_numpy(Y_nor).float().to(device)

        X_64 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_0, point_x_1, L, 64, model_up_0, Y_nor, eps1,
                          t, method=method1)
        X_64 = X_64.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
        X_64 /= tmp_nor
        out_ls.append(X_64)

        X_64 = torch.from_numpy(X_64).float().to(device)

        if model_up_1:
            print('Sup 64x64->128x128')
            X_128 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_1, point_x_2, L, 128, model_up_1, X_64,
                               eps2, t, method=method2)
            X_128 = X_128.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
            X_128 /= tmp_nor
            out_ls.append(X_128)
            X_128 = torch.from_numpy(X_128).float().to(device)

            if model_up_2:
                print('Sup 128x128->256x256')
                X_256 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_2, point_x, L, 256, model_up_2,
                                   X_128, eps3, t, method=method3)
                X_256 = X_256.detach().cpu().numpy()
                tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
                X_256 /= tmp_nor
                out_ls.append(X_256)

    return out_ls

def transfer_ddib_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, marginal_prob_std_fn_b, diffusion_coeff_fn_b, point_x_0, point_x_1, point_x_2, point_x, L, eps, t1, t2, t, X, model_coarse_gen, model_down_gen, model_up_0=None, model_up_1=None, model_up_2=None):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_DDIB(marginal_prob_std_fn_b, diffusion_coeff_fn_b, model_coarse_gen, model_down_gen, X, eps, t1, t2)

    out_ls.append(Y)

    if model_up_0:
        norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
        Y_nor = Y / norf
        Y_nor = torch.from_numpy(Y_nor).float().to(device)

        X_64 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_0, point_x_1, L, 64, model_up_0, Y_nor, eps,
                          t)
        X_64 = X_64.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
        X_64 /= tmp_nor
        out_ls.append(X_64)

        X_64 = torch.from_numpy(X_64).float().to(device)

        if model_up_1:
            print('Sup 64x64->128x128')
            X_128 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_1, point_x_2, L, 128, model_up_1, X_64,
                               eps, t)
            X_128 = X_128.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
            X_128 /= tmp_nor
            out_ls.append(X_128)
            X_128 = torch.from_numpy(X_128).float().to(device)

            if model_up_2:
                print('Sup 128x128->256x256')
                X_256 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_2, point_x, L, 256, model_up_2,
                                   X_128, eps, t)
                X_256 = X_256.detach().cpu().numpy()
                tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
                X_256 /= tmp_nor
                out_ls.append(X_256)

    return out_ls

def transfer_sot_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_0, point_x_1, point_x_2, point_x, L, eps, t, X, model_T, model_up_0=None, model_up_1=None, model_up_2=None):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_SOT(model_T, X, z_size=1)

    out_ls.append(Y)

    if model_up_0:
        norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
        Y_nor = Y / norf
        Y_nor = torch.from_numpy(Y_nor).float().to(device)

        X_64 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_0, point_x_1, L, 64, model_up_0, Y_nor, eps,
                          t)
        X_64 = X_64.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
        X_64 /= tmp_nor
        out_ls.append(X_64)

        X_64 = torch.from_numpy(X_64).float().to(device)

        if model_up_1:
            print('Sup 64x64->128x128')
            X_128 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_1, point_x_2, L, 128, model_up_1, X_64,
                               eps, t)
            X_128 = X_128.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
            X_128 /= tmp_nor
            out_ls.append(X_128)
            X_128 = torch.from_numpy(X_128).float().to(device)

            if model_up_2:
                print('Sup 128x128->256x256')
                X_256 = Sup_up_one(marginal_prob_std_fn_s, diffusion_coeff_fn_s, point_x_2, point_x, L, 256, model_up_2,
                                   X_128, eps, t)
                X_256 = X_256.detach().cpu().numpy()
                tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
                X_256 /= tmp_nor
                out_ls.append(X_256)

    return out_ls

