import numpy as np
import torch
from utils_2d.train_sc_utils import *
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Super resolution')
parser.add_argument('-type', '--type', type=str, metavar='', help='type of problem')
parser.add_argument('-smth', '--smth', type=str, metavar='', help='smooth kernel')
parser.add_argument('-flag', '--flag', type=int, metavar='', help='flag of sup')
parser.add_argument('-rep', '--rep', type=int, metavar='', help='resume epoch')
parser.add_argument('-nep', '--nep', type=int, metavar='', help='new stopping epoch')
args = parser.parse_args()

if __name__ == "__main__":
    if args.type:
        print('User defined problem')
        type = args.type
        smooth = args.smth
        flag = args.flag
        resume_epoch = args.rep
        new_stopping_epoch = args.nep
    else:
        print('Not define problem type, use default poentential vorticity generate skip point at 32x32')
        type = 'PV'
        smooth = 'cv1'
        flag = 2
        resume_epoch = 200
        new_stopping_epoch = 400

    print(type, ' Problem with flag ', flag, ' smooth ', smooth)

    if type == 'PV':
        from config.SC_config_pv import *
        from config.config_pv import *
    elif type == 'WD':
        from config.SC_config_wd import *
        from config.config_wd import *
    elif type == 'NS':
        from config.SC_config_ns import *
        from config.config_ns import *
    elif type == 'ELAS0':
        from config.SC_config_elas0 import *
        from config.config_elas0 import *
    elif type == 'ELAS1':
        from config.SC_config_elas1 import *
        from config.config_elas1 import *
    elif type == 'ELAS2':
        from config.SC_config_elas2 import *
        from config.config_elas2 import *

    with open(Super_data, 'rb') as ss:

        sp_0 = np.load(ss)
        sp_1 = np.load(ss)
        sp_2 = np.load(ss)

        up1_0 = np.load(ss)
        up1_1 = np.load(ss)
        up1_2 = np.load(ss)

        up2_0 = np.load(ss)
        up2_1 = np.load(ss)
        up2_2 = np.load(ss)

        ref = np.load(ss)


    if flag == 0:
        if smooth == 'sp':
            up_0, tup_0 = sp_0[:N_train_sup, ...], sp_0[N_train_sup:, ...]
            save_name = save_name_up_sp_0
            model = model_up_sp_0
        elif smooth == 'cv1':
            up_0, tup_0 = up1_0[:N_train_sup, ...], up1_0[N_train_sup:, ...]
            save_name = save_name_up_cv1_0
            model = model_up_cv1_0
        elif smooth == 'cv2':
            up_0, tup_0 = up2_0[:N_train_sup, ...], up2_0[N_train_sup:, ...]
            save_name = save_name_up_cv2_0
            model = model_up_cv2_0

        mat_up = up_0
        tmat_up = tup_0
        train = torch.from_numpy(mat_up).float()
        bs = 10
        test_cond = tmat_up[..., [1]]
        test_cond = torch.from_numpy(test_cond).float()
        test_real = tmat_up[..., [0]]
        N_c = int(Nx / 4)
        xx_c, yy_c = xx_1, yy_1
        xx_f, yy_f = xx_2, yy_2
        save_epoch = save_epoch_large
        batch_size = batch_size_large
    if flag == 1:
        if smooth == 'sp':
            up_1, tup_1 = sp_1[:N_train_sup, ...], sp_1[N_train_sup:, ...]
            save_name = save_name_up_sp_1
            model = model_up_sp_1
        elif smooth == 'cv1':
            up_1, tup_1 = up1_1[:N_train_sup, ...], up1_1[N_train_sup:, ...]
            save_name = save_name_up_cv1_1
            model = model_up_cv1_1
        elif smooth == 'cv2':
            up_1, tup_1 = up2_1[:N_train_sup, ...], up2_1[N_train_sup:, ...]
            save_name = save_name_up_cv2_1
            model = model_up_cv2_1

        mat_up = up_1
        tmat_up = tup_1
        train = torch.from_numpy(mat_up).float()
        bs = 10
        test_cond = tmat_up[..., [1]]
        test_cond = torch.from_numpy(test_cond).float()
        test_real = tmat_up[..., [0]]
        N_c = int(Nx / 2)
        xx_c, yy_c = xx_2, yy_2
        xx_f, yy_f = xx, yy
        save_epoch = save_epoch_small
        batch_size = batch_size_small
    elif flag == 2:
        if smooth == 'sp':
            up_2, tup_2 = sp_2[:N_train_sup, ...], sp_2[N_train_sup:, ...]
            save_name = save_name_up_sp_2
            model = model_up_sp_2
        elif smooth == 'cv1':
            up_2, tup_2 = up1_2[:N_train_sup, ...], up1_2[N_train_sup:, ...]
            save_name = save_name_up_cv1_2
            model = model_up_cv1_2
        elif smooth == 'cv2':
            up_2, tup_2 = up2_2[:N_train_sup, ...], up2_2[N_train_sup:, ...]
            save_name = save_name_up_cv2_2
            model = model_up_cv2_2

        mat_up = up_2
        tmat_up = tup_2
        train = torch.from_numpy(mat_up).float()
        bs = 10
        test_cond = tmat_up[..., [1]]
        test_cond = torch.from_numpy(test_cond).float()
        test_real = tmat_up[..., [0]]
        N_c = int(Nx)
        xx_c, yy_c = xx, yy
        xx_f, yy_f = xx, yy
        save_epoch = save_epoch_small
        batch_size = batch_size_small


    my_loss_func = loss_score_t_cond

    dataset = TensorDataset(train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cwd = os.getcwd()

    log_name = cwd + '/logs/' + save_name + '_log.txt'
    chkpts_base_name = cwd + '/mdls/' + save_name

    content = log_name
    mylogger(log_name, content)
    print(content)

    chkpts_name = chkpts_base_name + '_epoch_' + str(resume_epoch) + '_ckpt.pth'
    ckpt = torch.load(chkpts_name)
    model.load_state_dict(ckpt)

    optimizer = Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

    for epoch in tqdm(range(resume_epoch, new_stopping_epoch)):
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            a, x = x[0][..., [1]].to(device), x[0][..., [0]].to(device)

            loss = my_loss_func(model, x, a, marginal_prob_std_fn_s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        #if epoch % save_epoch == 0 and epoch>0:
        if epoch % 50 == 0 and epoch > 0:
            # Print the averaged training loss so far.
            content = 'at epoch: %d, Average Loss: %3f' % (
                epoch, avg_loss / num_items)
            mylogger(log_name, content)
            print(content)

            ### test sampling
            bs = 10
            cond0 = test_cond[:bs, ...].to(device)
            #cond0 = tmat[:bs, ...].to(device)

            sample_1 = gen_sample_cond(model, cond0, N_c, bs, 1, marginal_prob_std_fn_s, diffusion_coeff_fn_s, 'ode_solver_cond')

            sample_1 = sample_1.detach().cpu().numpy()

            sample_real = test_real[:bs, ...]

            ### compare coefficent
            error1 = get_relative_l2_error(sample_1, sample_real)

            content = 'coeff error at epoch: %d, error ode is: %3f' % (
                epoch, error1)
            mylogger(log_name, content)
            print(content)

            # ######### plot #########################
            # bs = 4
            # nrows = 3
            # fig1, ax = plt.subplots(nrows, bs, figsize=(bs * 2, nrows * 2))
            # for i in range(bs):
            #     ax[0, i].contourf(xx_c, yy_c, cond0[i, ..., 0].detach().cpu().numpy(), 36, cmap=cm.jet)
            #     ax[1, i].contourf(xx_c, yy_c, sample_1[i, ..., 0], 36, cmap=cm.jet)
            #     ax[2, i].contourf(xx_c, yy_c, sample_real[i, ..., 0], 36, cmap=cm.jet)
            # plt.show()
            # ###########################################################################
            if epoch>0:
                chkpts_name = chkpts_base_name + '_epoch_' + str(epoch) + '_ckpt.pth'
                torch.save(model.state_dict(), chkpts_name)