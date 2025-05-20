import matplotlib.pyplot as plt
import numpy as np
from config.SC_config_1d import *
from config.config_1d import *
from utils_1d.data_utils import *
from utils_1d.solver_utils import *
import argparse

parser = argparse.ArgumentParser(description='Super resolution')
parser.add_argument('-type', '--type', type=str, metavar='', help='type of problem')
args = parser.parse_args()

if __name__ == "__main__":
    type = args.type

    ## Load models
    load_epoch =5000
    chkpts_name_c_gen = cwd + '/mdls/' + save_name_c_gen + '_epoch_' + str(load_epoch) + '_ckpt.pth'
    ckpt_c = torch.load(chkpts_name_c_gen)
    model_c_gen.load_state_dict(ckpt_c)

    ### Load data
    with open(Data_name, 'rb') as ss:
        real = np.load(ss)
        god = np.load(ss)
        lw = np.load(ss)
        cd = np.load(ss)
        fft = np.load(ss)

    ######### define problem type ###############
    bs = 50
    np.random.seed(42)

    if type == 'white' or type == 'pink' or type == 'brown':
        rows_to_delete = []
        for i in range(real.shape[0]):
            if np.max(np.abs(real[i, ...])) < 1e-2:
                rows_to_delete.append(i)

        real = np.delete(real, rows_to_delete, axis=0)
    real = real[:bs, ...]

    if type == 'cd':
        test = cd[:bs, ...]
    elif type == 'lw':
        test = lw[:bs, ...]
    elif type == 'god':
        test = god[:bs, ...]
    elif type == 'fft':
        test = fft[:bs, ...]
    elif type == 'white':
        test = real[:bs, ...] + 0.1 * np.random.randn(*real[:bs, ...].shape)
    elif type == 'pink':
        test = real[:bs, ...] + 0.1 * generate_batch_pink_noise_1d(bs, real.shape[1])
    elif type == 'brown':
        test = real[:bs, ...] + 0.1 * generate_batch_pink_noise_1d(bs, real.shape[1], pinkness=2)

    bpd = np.zeros((7, 4, bs))
    ipd = np.zeros((7, 4, bs))

    T_ls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    for i, T in enumerate(T_ls):
        for metric in range(1, 5):
            if metric == 1:
                method = 'mmd'
                weight = False
                Method_name = 'MMD'
            elif metric == 2:
                method = 'spectral'
                weight = False
                Method_name = 'MELRu'
            elif metric == 3:
                method = 'spectral'
                weight = True
                Method_name = 'MELRw'
            elif metric == 4:
                method = 'w2'
                weight = False
                Method_name = 'W2'

            sigma = 0.01
            Ts = 0.01
            t_ls = np.linspace(Ts, T, 10)

            min, t = find_opt_t(test[..., None], real[..., None], marginal_prob_std_fn, t_ls, max_k=64, metric=method,
                                weight=weight, sigma=sigma)

            min, t1, t2 = find_inter_t(test[..., None], real[..., None], marginal_prob_std_fn, t_ls, max_k=64,
                                       metric=method, weight=weight, sigma=sigma)

            Y_test = torch.from_numpy(test[:bs, ..., None]).float().to(device)

            transfer_sdit_test_imb = transfer_sdit(marginal_prob_std_fn, diffusion_coeff_fn, model_c_gen, Y_test, eps=1e-5,
                                                   t1=t1, t2=t2)
            transfer_sdit_test_vani = transfer_sdit(marginal_prob_std_fn, diffusion_coeff_fn, model_c_gen, Y_test, eps=1e-5,
                                                    t1=t, t2=t)

            error_vani2ref_tvd = compute_TVD_vec(transfer_sdit_test_vani, real[..., None])
            error_imb2ref_tvd = compute_TVD_vec(transfer_sdit_test_imb, real[..., None])

            # print(error_imb2ref_tvd.shape, error_vani2ref_tvd.shape)

            bpd[i, metric - 1, ...] = error_vani2ref_tvd
            ipd[i, metric - 1, ...] = error_imb2ref_tvd

    bpd_mmd, bpd_melru, bpd_melrw, bpd_w2 = bpd[:, 0, ...], bpd[:, 1, ...], bpd[:, 2, ...], bpd[:, 3, ...]
    ipd_mmd, ipd_melru, ipd_melrw, ipd_w2 = ipd[:, 0, ...], ipd[:, 1, ...], ipd[:, 2, ...], ipd[:, 3, ...]

    mean_bpd_mmd = np.mean(bpd_mmd, axis=1)
    var_bpd_mmd = np.var(bpd_mmd, axis=1)

    mean_bpd_melru = np.mean(bpd_melru, axis=1)
    var_bpd_melru = np.var(bpd_melru, axis=1)

    mean_bpd_melrw = np.mean(bpd_melrw, axis=1)
    var_bpd_melrw = np.var(bpd_melrw, axis=1)

    mean_bpd_w2 = np.mean(bpd_w2, axis=1)
    var_bpd_w2 = np.var(bpd_w2, axis=1)

    ###########################################

    mean_ipd_mmd = np.mean(ipd_mmd, axis=1)
    var_ipd_mmd = np.var(ipd_mmd, axis=1)

    mean_ipd_melru = np.mean(ipd_melru, axis=1)
    var_ipd_melru = np.var(ipd_melru, axis=1)

    mean_ipd_melrw = np.mean(ipd_melrw, axis=1)
    var_ipd_melrw = np.var(ipd_melrw, axis=1)

    mean_ipd_w2 = np.mean(ipd_w2, axis=1)
    var_ipd_w2 = np.var(ipd_w2, axis=1)







