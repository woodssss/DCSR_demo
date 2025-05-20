from utils_1d.train_sc_utils import *
from utils_1d.data_utils import *

def find_inter_t(X, Y, marginal_prob_std_fn, t1_ls, metric='spectral', max_k = 16, weight=False, sigma=0.1):
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).float().to(device)

    min_value = float('inf')
    #min_value = 100000000000
    opt_t1 = 100
    opt_t2 = 100
    for t1 in t1_ls:
        t2_ls = np.linspace(2*t1, 5*t1, 10)
        for t2 in t2_ls:
            #print(t1, t2)
            latent_x = get_perturbed_x(X, marginal_prob_std_fn, t=t1)
            latent_y = get_perturbed_x(Y, marginal_prob_std_fn, t=t2)

            if metric=='spectral':
                k_vec, E_x =compute_energy_spectrum_average(latent_x.detach().cpu().numpy())
                _, E_y = compute_energy_spectrum_average(latent_y.detach().cpu().numpy())

                _, dis = compute_melr(E_x, E_y, max_k=max_k, weighted=weight)
            elif metric == 'mmd':
                dis = compute_mmd(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy(), sigma=sigma)
            elif metric == 'w2':
                dis = compute_w2(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy())

            #print(dis)

            if dis < min_value:
                min_value = dis
                opt_t1, opt_t2 = t1, t2

    return min_value, opt_t1, opt_t2

def find_opt_t(X, Y, marginal_prob_std_fn, t1_ls, metric='spectral', max_k = 16, weight=False, sigma=0.1):
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).float().to(device)

    min_value = float('inf')
    opt_t = 100
    for t1 in t1_ls:
        latent_x = get_perturbed_x(X, marginal_prob_std_fn, t=t1)
        latent_y = get_perturbed_x(Y, marginal_prob_std_fn, t=t1)

        if metric == 'spectral':
            k_vec, E_x = compute_energy_spectrum_average(latent_x.detach().cpu().numpy())
            _, E_y = compute_energy_spectrum_average(latent_y.detach().cpu().numpy())

            _, dis = compute_melr(E_x, E_y, max_k=max_k, weighted=weight)
        elif metric == 'mmd':
            dis = compute_mmd(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy(), sigma=sigma)
        elif metric == 'w2':
            dis = compute_w2(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy())

        if dis < min_value:
            min_value = dis
            opt_t = t1

    return min_value, opt_t