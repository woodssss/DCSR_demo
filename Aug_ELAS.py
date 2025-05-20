import matplotlib.pyplot as plt
import torch
from sympy.printing.pretty.pretty_symbology import line_width
from utils_2d.train_sc_utils import *
from utils_2d.Neural_solver_up import *
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    smth = 'cv1'
    load_epoch_gen = 2000
    type = 'ELAS0'
    bs = 20

    if type == 'PV':
        from config.SC_config_pv import  *
        from config.config_pv import *
    elif type == 'WD':
        from config.SC_config_wd import  *
        from config.config_wd import *
    elif type == 'NS':
        from config.SC_config_ns import  *
        from config.config_ns import *
    elif type == 'ELAS0':
        from config.SC_config_elas0 import  *
        from config.config_elas0 import *
    elif type == 'ELAS1':
        from config.SC_config_elas1 import  *
        from config.config_elas1 import *
    elif type == 'ELAS2':
        from config.SC_config_elas2 import  *
        from config.config_elas2 import *

    if smth == 'sp':
        ### Load models
        chkpts_name_sp = cwd + '/mdls/' + save_name_sp_gen + '_epoch_' + str(load_epoch_gen) + '_ckpt.pth'
        ckpt_sp = torch.load(chkpts_name_sp)
        model_gen.load_state_dict(ckpt_sp)

    elif smth == 'cv1':
        ### Load models
        chkpts_name_cv1 = cwd + '/mdls/' + save_name_cv1_gen + '_epoch_' + str(load_epoch_gen) + '_ckpt.pth'
        ckpt_cv1 = torch.load(chkpts_name_cv1)
        model_gen.load_state_dict(ckpt_cv1)

    elif smth == 'cv2':
        ### Load models
        chkpts_name_cv2 = cwd + '/mdls/' + save_name_cv2_gen + '_epoch_' + str(load_epoch_gen) + '_ckpt.pth'
        ckpt_cv2 = torch.load(chkpts_name_cv2)
        model_gen.load_state_dict(ckpt_cv2)

    ### load data
    with open(Gen_data, 'rb') as ss:
        sp_train = np.load(ss)
        cv1_train = np.load(ss)
        cv2_train = np.load(ss)

    with open(Test_data, 'rb') as ss:
        l1 = np.load(ss)
        l2 = np.load(ss)

        ref = np.load(ss)

        sp = np.load(ss)
        cv1 = np.load(ss)
        cv2 = np.load(ss)

    # print(l1.shape, l2.shape, ref.shape, sp.shape, cv1.shape, cv2.shape)
    # print(np.max(np.abs(l1)), np.max(np.abs(l2)))
    # zxc

    l1, l2 = make_image(l1), make_image(l2)


    if smth == 'sp':
        target = sp[:bs, ...]
        train = sp_train[:100, ...]
    elif smth == 'cv1':
        target = cv1[:bs, ...]
        train = cv1_train[:100, ...]
    elif smth == 'cv2':
        target = cv2[:bs, ...]
        train = cv2_train[:100, ...]

    ### low measurement
    cc = l1[..., None]

    c_0_up_1 = interp_pbc_2d_batch(points_x_1, points_x_0, L, cc)
    c_0_up_1 = make_image(c_0_up_1)

    c_0_up_2 = interp_pbc_2d_batch(points_x_2, points_x_0, L, cc)
    c_0_up_2 = make_image(c_0_up_2)

    c_0_up_f = interp_pbc_2d_batch(points_x, points_x_0, L, cc)
    c_0_up_f = make_image(c_0_up_f)

    ### automatically select t1 and t2
    t1_ls = np.linspace(0.001, 0.05, 20)

    min, t1, t2 = find_inter_t(train, cc, marginal_prob_std_fn_b, t1_ls)
    print(t1, t2)
    t1, t2 = 0.01, 0.05
    ### By SDE edit ################################################
    Y = torch.from_numpy(cc[:bs, ..., [0]]).float().to(device)

    transfer_sdit_0 = transfer_sdit(marginal_prob_std_fn_b, diffusion_coeff_fn_b, model_gen, Y, eps=1e-5, t1=t1, t2=t2)
    transfer_sdit_0 = make_image(transfer_sdit_0)

    error_c = np.mean(np.linalg.norm(target - cc[:bs, ..., [0]], axis=(1, 2)) / np.linalg.norm(target, axis=(1, 2)))
    error_sdeit = np.mean(np.linalg.norm(target - transfer_sdit_0, axis=(1, 2)) / np.linalg.norm(target, axis=(1, 2)))

    print(error_c, error_sdeit)

    images_top = []
    images_mid = []
    images_bottom = []
    bs = 5
    nrows = 3
    fig1, ax = plt.subplots(nrows, bs, figsize=(bs * 2, nrows * 2))
    for i in range(1, bs):
        img_top = ax[0, i].contourf(xx_0, yy_0, cc[i, ..., 0], 16, cmap=cm.jet)
        images_top.append(img_top)

        img_mid = ax[1, i].contourf(xx_0, yy_0, transfer_sdit_0[i, ..., 0], 16, cmap=cm.jet)
        images_mid.append(img_mid)

        img_bottom = ax[2, i].contourf(xx_0, yy_0, target[i, ..., 0], 16, cmap=cm.jet)
        images_bottom.append(img_bottom)

        #ax[3, i].contourf(xx, yy, ref[i * 2, ..., 0], 36, cmap=cm.jet)

    ax[0, 0].text(0.5, 0.5, r'$u^l$' + '\n' + 'error=' + f'{error_c:.3f}', fontsize=16, ha='center', va='center')
    ax[1, 0].text(0.5, 0.5, 'Purif' + '\n' + 'error=' + f'{error_sdeit:.3f}', fontsize=16, ha='center', va='center')
    ax[2, 0].text(0.5, 0.5, r'$\tilde{u}^c$', fontsize=16, ha='center', va='center')

    for axs in ax.flat:
        axs.set_xticks([])
        axs.set_yticks([])
        axs.axis('off')

    cbar_top = fig1.colorbar(images_top[0], ax=ax[0, :], orientation='vertical', fraction=0.1)
    cbar_mid = fig1.colorbar(images_mid[0], ax=ax[1, :], orientation='vertical', fraction=0.1)
    cbar_bottom = fig1.colorbar(images_bottom[0], ax=ax[2, :], orientation='vertical', fraction=0.1)

    plt.show()
