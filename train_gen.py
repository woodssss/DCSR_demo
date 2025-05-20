import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils_2d.train_sc_utils import *

parser = argparse.ArgumentParser(description='Diffusion Model at 32x32')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('-type', '--type', type=str, metavar='', help='type of example')
parser.add_argument('-smth', '--smth', type=str, metavar='', help='smooth kernel')
args = parser.parse_args()

if __name__ == "__main__":

    if args.type:
        print('User defined problem')
        type = args.type
        smooth = args.smth
    else:
        print('Not define problem type, use default poentential vorticity generate skip point at 32x32')
        type = 'PV'
        smooth = 'sp'

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

    with open(Gen_data, 'rb') as ss:
        sp = np.load(ss)
        cv1 = np.load(ss)
        cv2 = np.load(ss)

    if smooth == 'sp':
        print('Gen sp')
        train = torch.from_numpy(sp[..., [0]]).float()
        save_name = save_name_sp_gen
        model = model_sp_gen
    elif smooth == 'cv1':
        print('Gen cv1')
        train = torch.from_numpy(cv1[..., [0]]).float()
        save_name = save_name_cv1_gen
        model = model_cv1_gen
    elif smooth == 'cv2':
        print('Gen cv2')
        train = torch.from_numpy(cv2[..., [0]]).float()
        save_name = save_name_cv2_gen
        model = model_cv2_gen

    my_loss_func = loss_score_t

    dataset = TensorDataset(train)
    data_loader = DataLoader(dataset, batch_size=batch_size_large, shuffle=True)

    total_params = sum(p.numel() for p in model.parameters())

    cwd = os.getcwd()

    log_name = cwd + '/logs/' + save_name + '_log.txt'
    chkpts_base_name = cwd + '/mdls/' + save_name

    optimizer = Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

    for epoch in tqdm(range(n_epochs)):
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            x = x[0][..., [0]].to(device)
            loss = my_loss_func(model, x, marginal_prob_std_fn_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        if epoch % save_epoch_large == 0 and epoch > 0:
        #if epoch % 200 == 0 and epoch > 0:
            content = 'at epoch: %d, Average Loss: %3f' % (
                epoch, avg_loss / num_items)
            mylogger(log_name, content)
            print(content)

            # ### test sampling ###########################
            # bs = 8
            # mat = train[:bs, ...]
            # noise = torch.randn(bs, Nx_c, Nx_c, 1).to(device)
            # t = torch.ones(bs, device=device)
            #
            # noise *= marginal_prob_std_fn_b(t)[:, None, None, None]
            # sample_f = ode_solver(model, marginal_prob_std_fn_b, diffusion_coeff_fn_b, noise, forward=2, eps=1e-5)
            #
            # fig1, ax = plt.subplots(2, bs, figsize=(bs * 2, 4))
            # for i in range(bs):
            #     ax[0, i].contourf(xx_0, yy_0, sample_f[i, ..., 0].detach().cpu().numpy(), 36, cmap=cm.jet)
            #     ax[1, i].contourf(xx_0, yy_0, train[i, ..., 0].detach().cpu().numpy(), 36, cmap=cm.jet)
            # plt.show()
            # ###########################

            chkpts_name = chkpts_base_name + '_epoch_' + str(epoch) + '_ckpt.pth'
            torch.save(model.state_dict(), chkpts_name)