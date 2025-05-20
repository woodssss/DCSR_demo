import numpy as np
from SC_config import *
from config_cm import *
from Neural_solver_up import *
import copy
import seaborn as sns

if __name__ == "__main__":

    id = 1
    smooth = 'k2'
    res_level = 1

    load_epoch, load_epoch_up_0, load_epoch_up_1, load_epoch_up_2 = 2000, 2000, 1600, 200

    if smooth == 'k1':
        ### Load models
        chkpts_name_cv1 = cwd + '/mdls/' + save_name_cv1_gen + '_epoch_' + str(load_epoch) + '_ckpt.pth'
        ckpt_cv1 = torch.load(chkpts_name_cv1)
        model_cv1_gen.load_state_dict(ckpt_cv1)
        model_cv_gen = model_cv1_gen

        ######### For super-resolution ###############################
        chkpts_name_up_0 = cwd + '/mdls/' + save_name_up_cv1_0+ '_epoch_' + str(load_epoch_up_0) + '_ckpt.pth'
        ckpt_up_0 = torch.load(chkpts_name_up_0)
        model_up_cv1_0.load_state_dict(ckpt_up_0)
        model_up_cv_0 = model_up_cv1_0

        chkpts_name_up_1 = cwd + '/mdls/' + save_name_up_cv1_1 + '_epoch_' + str(load_epoch_up_1) + '_ckpt.pth'
        ckpt_up_1 = torch.load(chkpts_name_up_1)
        model_up_cv1_1.load_state_dict(ckpt_up_1)
        model_up_cv_1 = model_up_cv1_1

        if res_level>1:

            chkpts_name_up_2 = cwd + '/mdls/' + save_name_up_cv1_2 + '_epoch_' + str(load_epoch_up_2) + '_ckpt.pth'
            ckpt_up_2 = torch.load(chkpts_name_up_2)
            model_up_cv1_2.load_state_dict(ckpt_up_2)
            model_up_cv_2 = model_up_cv1_2
        #######################################################################################
    elif smooth == 'k2':
        ### Load models
        chkpts_name_cv2 = cwd + '/mdls/' + save_name_cv2_gen + '_epoch_' + str(load_epoch) + '_ckpt.pth'
        ckpt_cv2 = torch.load(chkpts_name_cv2)
        model_cv2_gen.load_state_dict(ckpt_cv2)
        model_cv_gen = model_cv2_gen

        ######### For super-resolution ###############################
        chkpts_name_up_0 = cwd + '/mdls/' + save_name_up_cv2_0 + '_epoch_' + str(load_epoch_up_0) + '_ckpt.pth'
        ckpt_up_0 = torch.load(chkpts_name_up_0)
        model_up_cv2_0.load_state_dict(ckpt_up_0)
        model_up_cv_0 = model_up_cv2_0


        chkpts_name_up_1 = cwd + '/mdls/' + save_name_up_cv2_1 + '_epoch_' + str(load_epoch_up_1) + '_ckpt.pth'
        ckpt_up_1 = torch.load(chkpts_name_up_1)
        model_up_cv2_1.load_state_dict(ckpt_up_1)
        model_up_cv_1 = model_up_cv2_1

        if res_level > 1:

            chkpts_name_up_2 = cwd + '/mdls/' + save_name_up_cv2_2 + '_epoch_' + str(load_epoch_up_2) + '_ckpt.pth'
            ckpt_up_2 = torch.load(chkpts_name_up_2)
            model_up_cv2_2.load_state_dict(ckpt_up_2)
            model_up_cv_2 = model_up_cv2_2
        #######################################################################################

    with open(Test_data, 'rb') as ss:
        c1 = np.load(ss)
        c2 = np.load(ss)
        c3 = np.load(ss)

        cv1 = np.load(ss)
        cv2 = np.load(ss)

        f = np.load(ss)

    with open(Super_test_data, 'rb') as ss:
        c1 = np.load(ss)
        c2 = np.load(ss)
        c3 = np.load(ss)

        up1_0 = np.load(ss)
        up1_1 = np.load(ss)
        up1_2 = np.load(ss)

        up2_0 = np.load(ss)
        up2_1 = np.load(ss)
        up2_2 = np.load(ss)

        ref = np.load(ss)

    if id==1:
        cc = c1.copy()
    elif id ==2:
        cc = c2.copy()
    else:
        cc = c3.copy()

    c_0_up_1 = interp_pbc_2d_batch(points_x_1, points_x_0, L, cc)
    c_0_up_1 = make_image(c_0_up_1)

    c_0_up_2 = interp_pbc_2d_batch(points_x_2, points_x_0, L, cc)
    c_0_up_2 = make_image(c_0_up_2)

    c_0_up_f = interp_pbc_2d_batch(points_x, points_x_0, L, cc)
    c_0_up_f = make_image(c_0_up_f)

    t1 = 0.01
    t2 = 0.05
    eps = 1e-5
    t = 1
    bs = 20

    c_ic = cc[:bs, ...]

    c_64 = interp_pbc_2d_batch(points_x_1, points_x_0, L, c_ic)
    c_64 = make_image(c_64)
    c_128 = interp_pbc_2d_batch(points_x_2, points_x_0, L, c_ic)
    c_128 = make_image(c_128)
    c_256 = interp_pbc_2d_batch(points_x, points_x_0, L, c_ic)
    c_256 = make_image(c_256)

    X = c_ic

    #### up to 128
    c_res_ls = direct_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, points_x_0, points_x_1, points_x_2, points_x,
                            L, eps, t, X, model_up_cv_0, model_up_1=model_up_cv_1, model_up_2=None)

    c_up_64, c_up_128 = c_res_ls[0], c_res_ls[1]

    sdit_res_ls = transfer_sdit_super(marginal_prob_std_fn_s, diffusion_coeff_fn_s, marginal_prob_std_fn_b,
                                      diffusion_coeff_fn_b, points_x_0, points_x_1, points_x_2, points_x, L, eps, t1,
                                      t2,
                                      t, X, model_cv_gen, model_up_cv_0, model_up_1=model_up_cv_1,
                                      model_up_2=None)
    sdit_32, sdit_64, sdit_128 = sdit_res_ls[0], sdit_res_ls[1], sdit_res_ls[2]

    real_128 = up1_1[:bs, ..., [0]]
    c_128 = c_128[:bs, ...]

    RMSE_c, covRMSE_c, TVD_c, melr_u_c, melr_w_c, k_vec, log_c = compute_all_error(c_128, real_128, max_k=64)
    RMSE_cup, covRMSE_cup, TVD_cup, melr_u_cup, melr_w_cup, k_vec, log_cup = compute_all_error(c_up_128, real_128,
                                                                                               max_k=64)
    RMSE_sdit, covRMSE_sdit, TVD_sdit, melr_u_sdit, melr_w_sdit, k_vec, log_sdit = compute_all_error(sdit_128, real_128,
                                                                                                     max_k=64)

    log_name = 'Error_res_id_' + str(id) + '_smooth_' + smooth + '.txt'

    content = 'c error ', RMSE_c, covRMSE_c, TVD_c, melr_u_c, melr_w_c
    mylogger(log_name, content)
    content = 'c up error ', RMSE_cup, covRMSE_cup, TVD_cup, melr_u_cup, melr_w_cup
    mylogger(log_name, content)
    content = 'sdit error ', RMSE_sdit, covRMSE_sdit, TVD_sdit, melr_u_sdit, melr_w_sdit
    mylogger(log_name, content)

    nrows, bs_plt = 4, 2
    fig1, ax = plt.subplots(bs_plt, nrows, figsize=(nrows * 2, bs_plt * 2))
    for i in range(bs_plt):
        ax[i, 0].contourf(xx_2, yy_2, c_128[i, ..., 0], 36, cmap=cm.jet)
        ax[i, 1].contourf(xx_2, yy_2, c_up_128[i, ..., 0], 36, cmap=cm.jet)
        ax[i, 2].contourf(xx_2, yy_2, sdit_128[i, ..., 0], 36, cmap=cm.jet)
        ax[i, 3].contourf(xx_2, yy_2, real_128[i, ..., 0], 36, cmap=cm.jet)

    ax[0, 0].set_title('Low interp', fontsize=12)
    ax[0, 1].set_title('coarse sup', fontsize=12)
    ax[0, 2].set_title('sdit sup', fontsize=12)
    ax[0, 3].set_title('ref', fontsize=12)

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])
        axs.axis('off')

    plt.figure(2)
    plt.plot(k_vec, log_c, linewidth=2, label='coarse up')
    plt.plot(k_vec, log_cup, linewidth=2, label='coarse sup')
    plt.plot(k_vec, log_sdit, linewidth=2, label='sdit sup')
    plt.legend()
    plt.xlabel('k', fontsize='20')
    plt.ylabel(r'$\| \log (E_k/E^{ref}_k) \|$', fontsize='20')
    plt.ylim([0, 3])

    plt.show()

    # tmp_file_name = 'PD_id_' + str(id) + '_smooth_' + smooth + '.npy'
    # with open(tmp_file_name, 'wb') as ss:
    #     np.save(ss, c_128)
    #     np.save(ss, c_up_128)
    #     np.save(ss, sdit_128)
    #     np.save(ss, real_128)


