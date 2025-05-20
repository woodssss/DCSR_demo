import torch
from utils_1d.train_sc_utils import *
from config.SC_config_1d import *
from config.config_1d import *
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import argparse

if __name__ == "__main__":
    with open(Data_name, 'rb') as ss:
        mat = np.load(ss)


    print('Gen coarse')
    train = torch.from_numpy(mat[..., None]).float()
    save_name = save_name_c_gen
    model = model_c_gen

    my_loss_func = loss_score_t

    dataset = TensorDataset(train)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    total_params = sum(p.numel() for p in model.parameters())

    cwd = os.getcwd()

    log_name = cwd + '/logs/' + save_name + '_log.txt'
    chkpts_base_name = cwd + '/mdls/' + save_name

    optimizer = Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

    for epoch in tqdm(range(5001)):
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            x = x[0][..., [0]].to(device)

            loss = my_loss_func(model, x, marginal_prob_std_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        if epoch % 500 == 0 and epoch > 0:
        #if epoch % 100 == 0 and epoch > 0:
            content = 'at epoch: %d, Average Loss: %3f' % (
                epoch, avg_loss / num_items)
            mylogger(log_name, content)
            print(content)

            # ### test sampling ###########################
            # bs = 8
            # mat = train[:bs, ...]
            # noise = torch.randn(bs, Nx, 1).to(device)
            # t = torch.ones(bs, device=device)
            #
            # noise *= marginal_prob_std_fn(t)[:, None, None]
            # sample_f = ode_solver(model, marginal_prob_std_fn, diffusion_coeff_fn, noise, forward=2, eps=1e-5)
            # print(sample_f.shape)
            #
            # fig1, ax = plt.subplots(2, bs, figsize=(bs * 2, 4))
            # for i in range(bs):
            #     ax[0, i].plot(points, sample_f[i, ..., 0].detach().cpu().numpy())
            #     ax[1, i].plot(points, train[i, ..., 0].detach().cpu().numpy())
            # plt.show()
            #
            # ###########################

            chkpts_name = chkpts_base_name + '_epoch_' + str(epoch) + '_ckpt.pth'
            torch.save(model.state_dict(), chkpts_name)
