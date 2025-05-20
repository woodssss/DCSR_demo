import matplotlib.pyplot as plt
import torch
from config.config_ns import *
from utils_2d.Neural_solver_up import *
import seaborn as sns
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Super resolution')
parser.add_argument('-problem', '--problem', type=int, metavar='', help='problem')
args = parser.parse_args()

if __name__ == "__main__":
    smth = 'cv2'
    if args.problem:
        problem = args.problem
    else:
        problem = 1

    load_epoch_gen, load_epoch_0, load_epoch_1= 2000, 2000, 2000
    load_epoch_2 = 350
    #load_epoch_2 = None

    if smth == 'sp':
        ### Load models
        chkpts_name_sp = cwd + '/mdls/' + save_name_sp_gen + '_epoch_' + str(load_epoch_gen) + '_ckpt.pth'
        ckpt_sp = torch.load(chkpts_name_sp)
        model_gen.load_state_dict(ckpt_sp)
        ######### For super-resolution ###############################
        chkpts_name_up_0 = cwd + '/mdls/' + save_name_up_sp_0 + '_epoch_' + str(load_epoch_0) + '_ckpt.pth'
        ckpt_up_0 = torch.load(chkpts_name_up_0)
        model_up_0.load_state_dict(ckpt_up_0)

        chkpts_name_up_1 = cwd + '/mdls/' + save_name_up_sp_1 + '_epoch_' + str(load_epoch_1) + '_ckpt.pth'
        ckpt_up_1 = torch.load(chkpts_name_up_1)
        model_up_1.load_state_dict(ckpt_up_1)

        if load_epoch_2:
            chkpts_name_up_2 = cwd + '/mdls/' + save_name_up_sp_2 + '_epoch_' + str(load_epoch_2) + '_ckpt.pth'
            ckpt_up_2 = torch.load(chkpts_name_up_2)
            model_up_2.load_state_dict(ckpt_up_2)

    elif smth == 'cv1':
        ### Load models
        chkpts_name_cv1 = cwd + '/mdls/' + save_name_cv1_gen + '_epoch_' + str(load_epoch_gen) + '_ckpt.pth'
        ckpt_cv1 = torch.load(chkpts_name_cv1)
        model_gen.load_state_dict(ckpt_cv1)
        ######### For super-resolution ###############################
        chkpts_name_up_0 = cwd + '/mdls/' + save_name_up_cv1_0 + '_epoch_' + str(load_epoch_0) + '_ckpt.pth'
        ckpt_up_0 = torch.load(chkpts_name_up_0)
        model_up_0.load_state_dict(ckpt_up_0)

        chkpts_name_up_1 = cwd + '/mdls/' + save_name_up_cv1_1 + '_epoch_' + str(load_epoch_1) + '_ckpt.pth'
        ckpt_up_1 = torch.load(chkpts_name_up_1)
        model_up_1.load_state_dict(ckpt_up_1)

        if load_epoch_2:
            chkpts_name_up_2 = cwd + '/mdls/' + save_name_up_cv1_2 + '_epoch_' + str(load_epoch_2) + '_ckpt.pth'
            ckpt_up_2 = torch.load(chkpts_name_up_2)
            model_up_2.load_state_dict(ckpt_up_2)

    elif smth == 'cv2':

        ### Load models
        chkpts_name_cv2 = cwd + '/mdls/' + save_name_cv2_gen + '_epoch_' + str(load_epoch_gen) + '_ckpt.pth'
        ckpt_cv2 = torch.load(chkpts_name_cv2)
        model_gen.load_state_dict(ckpt_cv2)
        ######### For super-resolution ###############################
        chkpts_name_up_0 = cwd + '/mdls/' + save_name_up_cv2_0 + '_epoch_' + str(load_epoch_0) + '_ckpt.pth'
        ckpt_up_0 = torch.load(chkpts_name_up_0)
        model_up_0.load_state_dict(ckpt_up_0)

        chkpts_name_up_1 = cwd + '/mdls/' + save_name_up_cv2_1 + '_epoch_' + str(load_epoch_1) + '_ckpt.pth'
        ckpt_up_1 = torch.load(chkpts_name_up_1)
        model_up_1.load_state_dict(ckpt_up_1)

        if load_epoch_2:
            chkpts_name_up_2 = cwd + '/mdls/' + save_name_up_cv2_2 + '_epoch_' + str(load_epoch_2) + '_ckpt.pth'
            ckpt_up_2 = torch.load(chkpts_name_up_2)
            model_up_2.load_state_dict(ckpt_up_2)

    ### load data
    with open(Gen_data, 'rb') as ss:
        sp_train = np.load(ss)
        cv1_train = np.load(ss)
        cv2_train = np.load(ss)

    with open(Test_data, 'rb') as ss:
        l1 = np.load(ss)
        l2 = np.load(ss)
        l3 = np.load(ss)

        ref = np.load(ss)

        sp = np.load(ss)
        cv1 = np.load(ss)
        cv2 = np.load(ss)

        sp_0 = np.load(ss)
        sp_1 = np.load(ss)
        sp_2 = np.load(ss)

        up1_0 = np.load(ss)
        up1_1 = np.load(ss)
        up1_2 = np.load(ss)

        up2_0 = np.load(ss)
        up2_1 = np.load(ss)
        up2_2 = np.load(ss)


    if smth == 'sp':
        target = sp
        train = sp_train[:100, ...]
    elif smth == 'cv1':
        target = cv1
        train = cv1_train[:100, ...]
    elif smth == 'cv2':
        target = cv2
        train = cv2_train[:100, ...]

    ### low measurement
    if problem == 1:
        cc = l1.copy()[..., None]
    elif problem == 2:
        cc = l2.copy()[..., None]
    elif problem == 3:
        cc = l3.copy()[..., None]

    cc = make_image(cc)

    c_0_up_1 = interp_pbc_2d_batch(points_x_1, points_x_0, L, cc)
    c_0_up_1 = make_image(c_0_up_1)

    c_0_up_2 = interp_pbc_2d_batch(points_x_2, points_x_0, L, cc)
    c_0_up_2 = make_image(c_0_up_2)

    c_0_up_f = interp_pbc_2d_batch(points_x, points_x_0, L, cc)
    c_0_up_f = make_image(c_0_up_f)

    ########## define parameter #################################
    bs = 16
    ### automatically select t1 and t2
    t1_ls = np.linspace(0.005, 0.05, 10)

    min, t1, t2 = find_inter_t(train, cc, marginal_prob_std_fn_b, t1_ls, cmin=4, cmax=8)
    t = 1

    # eps = 1e-6
    # method = 'ode_solver_cond'

    eps = 1e-5

    # eps1, eps2, eps3 = 1e-3, 1e-3, 1e-5
    # method1, method2, method3 = 'pc_sampler_cond', 'pc_sampler_cond', 'ode_solver_cond'

    # eps1, eps2, eps3 = 1e-5, 1e-3, 1e-3
    # method1, method2, method3 = 'ode_solver_cond', 'pc_sampler_cond', 'pc_sampler_cond'

    eps1, eps2, eps3 = 1e-5, 1e-5, 1e-5
    method1, method2, method3 = 'ode_solver_cond', 'ode_solver_cond', 'ode_solver_cond'

    # ### 32x32 -> 128x128 ################################################
    # ### By SDE edit
    # X = torch.from_numpy(cc[:bs, ..., [0]]).float().to(device)
    #
    # ls = transfer_sdit_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, marginal_prob_std_fn_b,
    #                             diffusion_coeff_fn_b, points_x_0, points_x_1, points_x_2, points_x, L, eps, t1, t2, t, X, model_down_gen=model_gen,
    #                             model_up_0=model_up_0, model_up_1=model_up_1, model_up_2=None, method1=method1, method2=method2, method3=method3, eps1=eps1, eps2=eps2, eps3=eps3)
    #
    # pd_32, pd_64, pd_128 = ls[0], ls[1], ls[2]
    # pd_128 = make_image(pd_128)
    #
    # ### By direct Super-resolution
    # c_res_ls = direct_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, points_x_0, points_x_1, points_x_2, points_x,
    #                         L, eps, t, X, model_up_0, model_up_1=model_up_1, model_up_2=None, method1=method1, method2=method2, method3=method3, eps1=eps1, eps2=eps2, eps3=eps3)
    #
    # c_up_64, c_up_128 = c_res_ls[0], c_res_ls[1]
    #
    # #### Compute error
    # real_128 = up1_1[:bs, ..., [0]]
    # c_128 = c_0_up_2[:bs, ...]
    #
    # RMSE_c, MMD_c, TVD_c, melr_u_c, melr_w_c, k_vec, log_c = compute_all_error(c_128, real_128, max_k=64)
    # RMSE_cup, MMD_cup, TVD_cup, melr_u_cup, melr_w_cup, k_vec, log_cup = compute_all_error(c_up_128, real_128,
    #                                                                                            max_k=64)
    # RMSE_sdit, MMD_sdit, TVD_sdit, melr_u_sdit, melr_w_sdit, k_vec, log_sdit = compute_all_error(pd_128, real_128,
    #                                                                                                  max_k=64)
    #
    # log_name = 'NS_Error_res_smooth_' + smth + '.txt'
    #
    # content = 'c error ', RMSE_c, MMD_c, TVD_c, melr_u_c, melr_w_c
    # mylogger(log_name, content)
    # content = 'c up error ', RMSE_cup, MMD_cup, TVD_cup, melr_u_cup, melr_w_cup
    # mylogger(log_name, content)
    # content = 'sdit error ', RMSE_sdit, MMD_sdit, TVD_sdit, melr_u_sdit, melr_w_sdit
    # mylogger(log_name, content)
    #
    # ### plot some result
    # nrows = 5
    # bs_plt = 4
    # fig1, ax = plt.subplots(nrows, bs_plt, figsize=(bs_plt * 2, nrows * 2))
    # for i in range(1, bs_plt):
    #     ax[0, i].imshow(cc[i, ..., 0], cmap='viridis')
    #     ax[1, i].imshow(pd_32[i, ..., 0], cmap='viridis')
    #     ax[2, i].imshow(pd_64[i, ..., 0], cmap='viridis')
    #     ax[3, i].imshow(pd_128[i, ..., 0], cmap='viridis')
    #     ax[4, i].imshow(ref[i, ..., 0], cmap='viridis')
    #
    # for axs in ax.flat:
    #     axs.set_xticks([])
    #     axs.set_yticks([])
    #     axs.axis('off')
    #
    # nrows, bs_plt = 4, 2
    # fig2, ax = plt.subplots(bs_plt, nrows, figsize=(nrows * 2, bs_plt * 2))
    # for i in range(bs_plt):
    #     ax[i, 0].contourf(xx_2, yy_2, c_128[i, ..., 0], 36, cmap=cm.jet)
    #     ax[i, 1].contourf(xx_2, yy_2, c_up_128[i, ..., 0], 36, cmap=cm.jet)
    #     ax[i, 2].contourf(xx_2, yy_2, pd_128[i, ..., 0], 36, cmap=cm.jet)
    #     ax[i, 3].contourf(xx_2, yy_2, real_128[i, ..., 0], 36, cmap=cm.jet)
    #
    # ax[0, 0].set_title('Low interp', fontsize=12)
    # ax[0, 1].set_title('coarse sup', fontsize=12)
    # ax[0, 2].set_title('sdit sup', fontsize=12)
    # ax[0, 3].set_title('ref', fontsize=12)
    #
    # for axs in ax.flat:
    #     axs.set_xticks([])
    #     axs.set_yticks([])
    #     axs.axis('off')
    #
    # plt.figure(3)
    # plt.plot(k_vec, log_c, linewidth=2, label='coarse up')
    # plt.plot(k_vec, log_cup, linewidth=2, label='coarse sup')
    # plt.plot(k_vec, log_sdit, linewidth=2, label='sdit sup')
    # plt.legend()
    # plt.xlabel('k', fontsize='20')
    # plt.ylabel(r'$\| \log (E_k/E^{ref}_k) \|$', fontsize='20')
    # plt.xlim([0, 80])
    # plt.ylim([0, 6])
    #
    # plt.show()
    # ################################################################################################

    ### 32x32 -> 256x256 ################################################
    ### By SDE edit
    X = torch.from_numpy(cc[:bs, ..., [0]]).float().to(device)

    ls = transfer_sdit_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, marginal_prob_std_fn_b,
                             diffusion_coeff_fn_b, points_x_0, points_x_1, points_x_2, points_x, L, eps, t1, t2, t, X,
                             model_down_gen=model_gen,
                             model_up_0=model_up_0, model_up_1=model_up_1, model_up_2=model_up_2, method1=method1, method2=method2, method3=method3, eps1=eps1, eps2=eps2, eps3=eps3)

    pd_32, pd_64, pd_128, pd_256 = ls[0], ls[1], ls[2], ls[3]
    pd_256 = make_image(pd_256)

    ### By direct Super-resolution
    c_res_ls = direct_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, points_x_0, points_x_1, points_x_2, points_x,
                            L, eps, t, X, model_up_0, model_up_1=model_up_1, model_up_2=model_up_2, method1=method1, method2=method2, method3=method3, eps1=eps1, eps2=eps2, eps3=eps3)

    c_up_64, c_up_128, c_up_256 = c_res_ls[0], c_res_ls[1], c_res_ls[2]

    #### Compute error
    real_256 = up2_2[:bs, ..., [0]]
    c_256 = c_0_up_f[:bs, ...]

    RMSE_c, MMD_c, TVD_c, melr_u_c, melr_w_c, k_vec, log_c = compute_all_error(c_256, real_256, max_k=80)
    RMSE_cup, MMD_cup, TVD_cup, melr_u_cup, melr_w_cup, k_vec, log_cup = compute_all_error(c_up_256, real_256,
                                                                                               max_k=80)
    RMSE_sdit, MMD_sdit, TVD_sdit, melr_u_sdit, melr_w_sdit, k_vec, log_sdit = compute_all_error(pd_256, real_256,
                                                                                                     max_k=80)

    cwd = os.getcwd()
    log_name = cwd + '/figs/NS_Error_res_problem_' + str(problem) + '.txt'
    fig_name_process = cwd + '/figs/NS_proces_problem_' + str(problem) + '.png'
    fig_name_ins = cwd + '/figs/NS_ins_problem_problem_' + str(problem) + '.png'
    fig_name_spectral = cwd + '/figs/NS_spectral_problem_problem_' + str(problem) + '.png'
    npy_name = cwd + '/figs/NS_problem_' + str(problem) + '.npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, cc)
        np.save(ss, pd_32)
        np.save(ss, pd_64)
        np.save(ss, pd_128)
        np.save(ss, pd_256)
        np.save(ss, c_256)
        np.save(ss, c_up_256)
        np.save(ss, ref)

    content = 'c error ', RMSE_c, MMD_c, TVD_c, melr_u_c, melr_w_c
    mylogger(log_name, content)
    content = 'c up error ', RMSE_cup, MMD_cup, TVD_cup, melr_u_cup, melr_w_cup
    mylogger(log_name, content)
    content = 'sdit error ', RMSE_sdit, MMD_sdit, TVD_sdit, melr_u_sdit, melr_w_sdit
    mylogger(log_name, content)

    ### plot some result
    nrows = 6
    bs_plt = 4
    fig1, ax = plt.subplots(nrows, bs_plt, figsize=(bs_plt * 2, nrows * 2))
    for i in range(1, bs_plt):
        ax[0, i].imshow(cc[i, ..., 0], cmap='jet')
        ax[1, i].imshow(pd_32[i, ..., 0], cmap='jet')
        ax[2, i].imshow(pd_64[i, ..., 0], cmap='jet')
        ax[3, i].imshow(pd_128[i, ..., 0], cmap='jet')
        ax[4, i].imshow(pd_256[i, ..., 0], cmap='jet')
        ax[5, i].imshow(ref[i, ..., 0], cmap='jet')

    ax[0, 0].text(0.5, 0.5, 'Low', fontsize=16, ha='center', va='center')
    ax[1, 0].text(0.5, 0.5, 'Purif', fontsize=16, ha='center', va='center')
    ax[2, 0].text(0.5, 0.5, 'Purif' + '\n' + '64x64', fontsize=16, ha='center', va='center')
    ax[3, 0].text(0.5, 0.5, 'Purif' + '\n' + '128x128', fontsize=16, ha='center', va='center')
    ax[4, 0].text(0.5, 0.5, 'Purif' + '\n' + '256x256', fontsize=16, ha='center', va='center')
    ax[5, 0].text(0.5, 0.5, 'Ref', fontsize=16, ha='center', va='center')

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])
        axs.axis('off')

    fig1.savefig(fig_name_process)

    nrows, bs_plt = 5, 2
    fig2, ax = plt.subplots(bs_plt, nrows, figsize=(nrows * 2, bs_plt * 2))
    for i in range(bs_plt):
        # ax[i, 0].contourf(xx_0, yy_0, cc[i, ..., 0], 12, cmap=cm.jet)
        # ax[i, 1].contourf(xx, yy, c_256[i, ..., 0], 12, cmap=cm.jet)
        # ax[i, 2].contourf(xx, yy, c_up_256[i, ..., 0], 12, cmap=cm.jet)
        # ax[i, 3].contourf(xx, yy, pd_256[i, ..., 0], 12, cmap=cm.jet)
        # ax[i, 4].contourf(xx, yy, real_256[i, ..., 0], 12, cmap=cm.jet)
        ax[i, 0].imshow(cc[i, ..., 0], cmap=cm.jet)
        ax[i, 1].imshow(c_256[i, ..., 0], cmap=cm.jet)
        ax[i, 2].imshow(c_up_256[i, ..., 0], cmap=cm.jet)
        ax[i, 3].imshow(pd_256[i, ..., 0], cmap=cm.jet)
        ax[i, 4].imshow(real_256[i, ..., 0], cmap=cm.jet)

    ax[0, 0].set_title('Low', fontsize=20)
    ax[0, 1].set_title('Low+Interp', fontsize=20)
    ax[0, 2].set_title('Low+Sup', fontsize=20)
    ax[0, 3].set_title('Purif+Sup', fontsize=20)
    ax[0, 4].set_title('Reference', fontsize=20)

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])
        axs.axis('off')

    fig2.savefig(fig_name_ins)

    plt.figure(3)
    plt.plot(k_vec, log_c, linewidth=2, label='Low+Interp')
    plt.plot(k_vec, log_cup, linewidth=2, label='Low+Sup')
    plt.plot(k_vec, log_sdit, linewidth=2, label='Purif+Sup')
    plt.legend(loc='upper right', fontsize=16)
    plt.xlabel('k', fontsize='20')
    plt.ylabel(r'$\| \log (E_k/E^{ref}_k) \|$', fontsize='20')
    plt.xlim([0, 80])
    plt.ylim([0, 6])

    plt.savefig(fig_name_spectral)

    plt.show()
    ################################################################################################

