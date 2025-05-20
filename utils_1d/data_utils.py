import torch
import math
import numpy as np
import pywt
from scipy import sparse
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray
import cv2
import scipy.stats as stats
import ot


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def num2str_deciaml(x):
    s = str(x)
    c = ''
    for i in range(len(s)):
        if s[i] == '0':
            c = c + 'z'
        elif s[i] == '.':
            c = c + 'p'
        elif s[i] == '-':
            c = c + 'n'
        else:
            c = c + s[i]

    return c

def tensor2nump(x):
    return x.cpu().detach().numpy()

def make_tensor(*args):
    return [torch.from_numpy(arg).float().to(device) for arg in args]

def make_image(mat):
    for i in range(mat.shape[0]):
        mat[i, ...] /= np.max(np.abs(mat[i, ...]))
    return mat

def make_image_one(mat):
    return mat / np.max(np.abs(mat))

def prepare_wave_one(mat, wavelet='haar'):
    w_0_ls = []

    N = mat.shape[0]

    #print('prepare wave')
    for i in tqdm(range(N)):

        #coeffs2 = pywt.dwt2(mat[i, ..., 0], wavelet)  # 'haar' wavelet is used here

        coeffs2 = pywt.wavedec2(mat[i, ..., 0], wavelet, level=1, mode='periodization')  # 'haar' wavelet is used here
        cA2, (cH2, cV2, cD2) = coeffs2
        mat2 = np.concatenate([cA2[None, ..., None], cH2[None, ..., None], cV2[None, ..., None], cD2[None, ..., None]],
                              axis=-1)

        w_0_ls.append(mat2)

    mat_w_0 = np.concatenate(w_0_ls, axis=0)
    ##################################################

    return mat_w_0

def prepare_wave(mat, wavelet='haar'):
    w_ng_ls = []
    w_0_ls = []
    w_1_ls = []
    w_2_ls = []

    N = mat.shape[0]

    #print('prepare wave zzz')
    for i in tqdm(range(N)):

        coeffs2 = pywt.dwt2(mat[i, ..., 0], wavelet)  # 'haar' wavelet is used here
        cA2, (cH2, cV2, cD2) = coeffs2
        mat2 = np.concatenate([cA2[None, ..., None], cH2[None, ..., None], cV2[None, ..., None], cD2[None, ..., None]],
                              axis=-1)


        coeffs1 = pywt.dwt2(cA2, wavelet)  # 'haar' wavelet is used here
        cA1, (cH1, cV1, cD1) = coeffs1
        mat1 = np.concatenate([cA1[None, ..., None], cH1[None, ..., None], cV1[None, ..., None], cD1[None, ..., None]],
                              axis=-1)


        coeffs0 = pywt.dwt2(cA1, wavelet)  # 'haar' wavelet is used here
        cA0, (cH0, cV0, cD0) = coeffs0
        mat0 = np.concatenate([cA0[None, ..., None], cH0[None, ..., None], cV0[None, ..., None], cD0[None, ..., None]],
                              axis=-1)

        coeffsng = pywt.dwt2(cA0, wavelet)  # 'haar' wavelet is used here
        cAng, (cHng, cVng, cDng) = coeffsng
        matng = np.concatenate([cAng[None, ..., None], cHng[None, ..., None], cVng[None, ..., None], cDng[None, ..., None]],
                              axis=-1)

        w_ng_ls.append(matng)
        w_0_ls.append(mat0)
        w_1_ls.append(mat1)
        w_2_ls.append(mat2)

    mat_w_2 = np.concatenate(w_2_ls, axis=0)
    mat_w_1 = np.concatenate(w_1_ls, axis=0)
    mat_w_0 = np.concatenate(w_0_ls, axis=0)
    mat_w_ng = np.concatenate(w_ng_ls, axis=0)
    ##################################################

    return mat_w_ng, mat_w_0, mat_w_1, mat_w_2

def prepare_cv_one(kernel, blur, mat):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    u_2 = np.zeros((N, int(Nx / 2), int(Nx / 2), 1))

    for i in tqdm(range(N)):
        tmp_2 = cv2.resize(mat[i, ..., 0], (int(Nx / 2), int(Nx / 2)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_2 = cv2.blur(tmp_2, (kernel, kernel))

        u_2[i, ..., 0] = tmp_2

    return u_2

def prepare_cv_data(kernel, blur, mat):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    u_0 = np.zeros((N, int(Nx / 8), int(Nx / 8), 1))  ## last dim 0 is ref and 1 is interp
    u_1 = np.zeros((N, int(Nx / 4), int(Nx / 4), 1))
    u_2 = np.zeros((N, int(Nx / 2), int(Nx / 2), 1))

    print('prepare up')
    for i in tqdm(range(N)):

        tmp_0 = cv2.resize(mat[i, ..., 0], (int(Nx / 8), int(Nx / 8)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_0 = cv2.blur(tmp_0, (kernel, kernel))

        u_0[i, ..., 0] = tmp_0

        tmp_1 = cv2.resize(mat[i, ..., 0], (int(Nx / 4), int(Nx / 4)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_1 = cv2.blur(tmp_1, (kernel, kernel))

        u_1[i, ..., 0] = tmp_1

        tmp_2 = cv2.resize(mat[i, ..., 0], (int(Nx / 2), int(Nx / 2)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_2 = cv2.blur(tmp_2, (kernel, kernel))

        u_2[i, ..., 0] = tmp_2

    return u_0, u_1, u_2


def prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, mat, kernel = 2, blur = None):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    mat_u_0 = np.zeros((N, int(Nx / 4), int(Nx / 4), 2))
    mat_u_1 = np.zeros((N, int(Nx / 2), int(Nx / 2), 2))
    mat_u_2 = np.zeros((N, Nx, Nx, 2))
    mat_u_d = np.zeros((N, Nx, Nx, 2))

    print('prepare up')
    for i in tqdm(range(N)):

        tmp_0 = cv2.resize(mat[i, ...], (int(Nx / 8), int(Nx / 8)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_0 = cv2.blur(tmp_0, (kernel, kernel))

        tmp_1 = cv2.resize(mat[i, ...], (int(Nx / 4), int(Nx / 4)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_1 = cv2.blur(tmp_1, (kernel, kernel))

        tmp_2 = cv2.resize(mat[i, ...], (int(Nx / 2), int(Nx / 2)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_2 = cv2.blur(tmp_2, (kernel, kernel))

        u0 = interp_pbc_2d(points_x_1, points_x_0, L, tmp_0)
        u1 = interp_pbc_2d(points_x_2, points_x_1, L, tmp_1)
        u2 = interp_pbc_2d(points_x, points_x_2, L, tmp_2)
        ud = interp_pbc_2d(points_x, points_x_0, L, tmp_0)

        mat_u_0[i, ..., 0] = make_image_one(tmp_1)
        mat_u_0[i, ..., 1] = make_image_one(u0)
        #

        mat_u_1[i, ..., 0] = make_image_one(tmp_2)
        mat_u_1[i, ..., 1] = make_image_one(u1)

        mat_u_2[i, ..., 0] = make_image_one(mat[i, ..., 0])
        mat_u_2[i, ..., 1] = make_image_one(u2)

        mat_u_d[i, ..., 0] = make_image_one(mat[i, ..., 0])
        mat_u_d[i, ..., 1] = make_image_one(ud)

    return mat_u_0, mat_u_1, mat_u_2, mat_u_d

def prepare_up_skip(L, points_x_0, points_x_1, points_x_2, points_x, mat):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    mat_u_0 = np.zeros((N, int(Nx / 4), int(Nx / 4), 2))  ## last dim 0 is ref and 1 is interp
    mat_u_1 = np.zeros((N, int(Nx / 2), int(Nx / 2), 2))
    mat_u_2 = np.zeros((N, Nx, Nx, 2))
    mat_u_d = np.zeros((N, Nx, Nx, 2))

    print('prepare up')
    for i in tqdm(range(N)):
        tmp_0 = mat[i, ::8, ::8, 0]
        tmp_1 = mat[i, ::4, ::4, 0]
        tmp_2 = mat[i, ::2, ::2, 0]

        u0 = interp_pbc_2d(points_x_1, points_x_0, L, tmp_0)
        u1 = interp_pbc_2d(points_x_2, points_x_1, L, tmp_1)
        u2 = interp_pbc_2d(points_x, points_x_2, L, tmp_2)
        ud = interp_pbc_2d(points_x, points_x_0, L, tmp_0)

        mat_u_0[i, ..., 0] = tmp_1
        mat_u_0[i, ..., 1] = u0
        #

        mat_u_1[i, ..., 0] = tmp_2
        mat_u_1[i, ..., 1] = u1

        mat_u_2[i, ..., 0] = mat[i, ..., 0]
        mat_u_2[i, ..., 1] = u2

        mat_u_d[i, ..., 0] = mat[i, ..., 0]
        mat_u_d[i, ..., 1] = ud

    return mat_u_0, mat_u_1, mat_u_2, mat_u_d


def interp_pbc_1d(x_new, x, L, f):
    x = np.concatenate([x, np.ones((1))*L], axis=0)
    f = np.concatenate([f, np.ones((1)) * f[0]], axis=0)
    func = CubicSpline(x, f)

    f = func(x_new)
    return f

def interp_pbc_2d(x_new, x, L, f):
    Nx = x.shape[0]
    Nx_new = x_new.shape[0]

    f_f_1 = np.zeros((Nx, Nx_new))
    f_f_2 = np.zeros((Nx_new, Nx_new))

    for i in range(Nx):
        f_f_1[i, :] = interp_pbc_1d(x_new, x, L, f[i, :])

    for j in range(Nx_new):
        f_f_2[:, j] = interp_pbc_1d(x_new, x, L, f_f_1[:, j])

    return f_f_2

def interp_pbc_2d_batch(x_new, x, L, f_mat):
    bs = f_mat.shape[0]
    Nx_new = x_new.shape[0]

    f_new_mat = np.zeros((bs, Nx_new, Nx_new, 1))

    for i in range(bs):
        f_new_mat[i, ..., 0] = interp_pbc_2d(x_new, x, L, f_mat[i, ..., 0])

    return f_new_mat


def generate_batch_pink_noise_1d(bs, N, scale=1.0, pinkness=1.0):
    """
    Generate a batch of 1D pink noise with shape [bs, N, 1].

    Parameters:
        bs (int): Batch size.
        N (int): Number of elements in each 1D noise array.
        scale (float): Scaling factor for the noise intensity.
        pinkness (float): Controls the degree of frequency scaling (1.0 for true pink noise).

    Returns:
        numpy.ndarray: Pink noise array with shape [bs, N, 1].
    """
    pink_noise_batch = []
    for _ in range(bs):
        # Frequency grid
        f = np.fft.fftfreq(N)
        f[0] = 1e-10  # Avoid division by zero at DC component

        # Generate random phase and amplitude
        phase = np.random.uniform(0, 2 * np.pi, N)
        amplitude = np.random.normal(size=N) + 1j * np.random.normal(size=N)

        # Scale spectrum by 1/f^pinkness
        spectrum = amplitude / (np.abs(f) ** pinkness)
        spectrum *= np.exp(1j * phase)

        # Inverse FFT to spatial domain
        pink_noise = np.fft.ifft(spectrum).real

        # Normalize and scale
        pink_noise -= pink_noise.mean()
        pink_noise /= pink_noise.std()
        pink_noise *= scale  # Apply scaling factor

        # Append to batch with an additional channel dimension
        pink_noise_batch.append(pink_noise)

    return np.array(pink_noise_batch)


def get_grid(Nx, L):
 dx = L/Nx
 points = np.linspace(0, L - dx, Nx)
 return points


def batch_iwt(Nx, ca, coeff):
    bs = ca.shape[0]
    res = np.zeros((bs, Nx, Nx, 1))
    for i in range(bs):
        tmp_coeff = (ca[i, ..., 0], (coeff[i, ..., 0], coeff[i, ..., 1], coeff[i, ..., 2]))
        res[i, ..., 0] = pywt.idwt2(tmp_coeff, wavelet='haar')

    return res

def recover_wave(Nf, ca, coeff):
    return batch_iwt(Nf, ca, coeff)

def down_sample_cv(mat, Nx, kernel=2, blur=None):
    bs = mat.shape[0]
    down = np.zeros((bs, Nx, Nx, 1))
    for i in range(bs):
        down[i, ..., 0] = cv2.resize(mat[i, ...], (Nx, Nx), interpolation=cv2.INTER_LINEAR)
        if blur:
            down[i, ..., 0] = cv2.blur(down[i, ..., 0], (kernel, kernel))

    return down

def down_sample_evo_cv(mat, Nx, kernel=2, blur=None):
    bs = mat.shape[0]
    T = mat.shape[1]
    down = np.zeros((bs, T, Nx, Nx, 1))
    for i in range(bs):
        for j in range(T):
            down[i, j, ..., 0] = cv2.resize(mat[i, j, ...], (Nx, Nx), interpolation=cv2.INTER_LINEAR)
            if blur:
                down[i, j, ..., 0] = cv2.blur(down[i, j, ..., 0], (kernel, kernel))

    return down

def get_relative_l2_error(pd, ref):
    return np.linalg.norm(pd - ref) / np.linalg.norm(ref)

def rbf_kernel(X, Y, sigma):
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2 * np.dot(X, Y.T)
    return np.exp(-distances / (2 * sigma ** 2))

def compute_mmd(X, Y, sigma=0.5):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples X and Y using an RBF kernel.

    Parameters:
        X (np.ndarray): First sample, shape (n, d), where n is the number of points and d is the dimensionality.
        Y (np.ndarray): Second sample, shape (m, d), where m is the number of points.
        sigma (float): Bandwidth of the RBF kernel.

    Returns:
        float: The MMD statistic.
    """

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    # Compute the kernels
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)

    # Compute the MMD statistic
    m = X.shape[0]
    n = Y.shape[0]

    term_XX = np.sum(K_XX) / (m * m)
    term_YY = np.sum(K_YY) / (n * n)
    term_XY = 2 * np.sum(K_XY) / (m * n)

    mmd = term_XX + term_YY - term_XY
    return mmd

def compute_energy_spectrum_one(signal, T=1):
    # Number of samples
    N = signal.shape[0]
    # Fast Fourier Transform (FFT)
    fft_result = np.fft.fft(signal)
    # Calculate the energy spectrum (power per frequency)
    #energy_spectrum = (2.0 / N) * np.abs(fft_result[:N // 2]) ** 2
    energy_spectrum = (1 / 2) * np.abs(fft_result[:N // 2]) ** 2
    # Corresponding frequency bins
    frequencies = np.fft.fftfreq(N, T)[:N // 2]
    return frequencies, energy_spectrum

def compute_energy_spectrum_average(mat):
    bs = mat.shape[0]
    N = mat.shape[1]

    E_k = np.zeros((N//2))

    for i in range(bs):
        k_vec, E_k_tmp = compute_energy_spectrum_one(mat[i, ..., 0])

        E_k += E_k_tmp

    return k_vec, E_k/bs

def compute_TVD(pd, ref):
    return np.mean(np.sum(np.abs(pd - ref), axis=(1,2)) / np.sum(np.abs(ref), axis=(1,2)))


def compute_melr(E_pred, E_ref, max_k=50, weighted=False):
    # Total number of modes (k)
    num_modes = max_k

    # Unweighted or Weighted weights
    if weighted:
        weights = E_ref / np.sum(E_ref)  # Weighted case: energy-based weights
    else:
        weights = np.ones_like(E_ref) # Unweighted case: equal weights

    # Compute MELR
    vec = np.abs(np.log(E_pred / E_ref))
    melr = np.sum(weights**2 * np.abs(np.log(E_pred / E_ref)))

    return vec, melr

def rbf_kernel(X, Y, sigma):
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2 * np.dot(X, Y.T)
    return np.exp(-distances / (2 * sigma ** 2))

def compute_cov_rmse(true_data, pred_data):
    bs = true_data.shape[0]
    # Step 1: Flatten the 32x32 matrices into vectors of shape [100, 1024]
    true_data_flattened = true_data.reshape(bs, -1)  # shape [100, 1024]
    pred_data_flattened = pred_data.reshape(bs, -1)  # shape [100, 1024]

    # Step 2: Compute covariance matrices
    true_cov = np.cov(true_data_flattened, rowvar=False)  # shape [1024, 1024]
    pred_cov = np.cov(pred_data_flattened, rowvar=False)  # shape [1024, 1024]

    # Step 3: Compute element-wise squared difference
    cov_diff = (true_cov - pred_cov) ** 2

    # Step 4: Compute mean of squared differences
    mean_squared_diff = np.mean(cov_diff)

    # Step 5: Compute square root of mean squared difference (covRMSE)
    cov_rmse = np.sqrt(mean_squared_diff)

    return cov_rmse

def compute_w2(pred, ref):
    """
    Compute the Wasserstein-1 distance between two empirical distributions
    in multidimensional space using optimal transport.

    Parameters:
    - batch_1 (numpy.ndarray): Samples from the first distribution, shape (n_samples_1, n_features).
    - batch_2 (numpy.ndarray): Samples from the second distribution, shape (n_samples_2, n_features).

    Returns:
    - float: The Wasserstein-1 distance between the two distributions.
    """
    pred = pred[..., 0]
    ref = ref[..., 0]
    # Compute pairwise cost matrix (Euclidean distances)
    cost_matrix = ot.dist(pred, ref, metric='euclidean')

    # Uniform weights for each distribution
    weights_1 = np.ones(len(pred)) / len(ref)
    weights_2 = np.ones(len(pred)) / len(ref)

    # Compute the squared Wasserstein distance
    res = ot.emd2(weights_1, weights_2, cost_matrix)

    return res

def compute_all_error(pd, ref, max_k):
    RMSE = get_relative_l2_error(pd, ref)
    covRMSE = compute_cov_rmse(pd, ref)
    TVD = compute_TVD(pd, ref)

    k_vec, E_pd = compute_energy_spectrum_average(pd, max_k)
    k_vec, E_ref = compute_energy_spectrum_average(ref, max_k)

    log_vec, melr_u = compute_melr(E_pd, E_ref, max_k, weighted=False)
    log_vec, melr_w = compute_melr(E_pd, E_ref, max_k, weighted=True)


    return RMSE, covRMSE, TVD, melr_u, melr_w, k_vec, log_vec









def compute_TVD_vec(pd, ref):
    return np.sum(np.abs(pd - ref), axis=(1,2)) / np.sum(np.abs(ref), axis=(1,2))





