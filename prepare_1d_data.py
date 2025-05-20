import numpy as np
from utils_1d.Numerical_solver import *
from config.config_1d import *
from utils_1d.GRF import *

if __name__ == "__main__":

    ic_mat = generate_sharp_grf(Ns, points, l)

    M_cd = central_difference_matrix(Nx, Lx / Nx, v, dt)
    M_uw = upwind_matrix(Nx, Lx / Nx, v, dt)
    M_lw = lax_wendroff_matrix(Nx, Lx / Nx, v, dt)
    M_lf = lax_friedrichs_matrix(Nx, Lx / Nx, v, dt)
    M_god = godunov_matrix(Nx, Lx / Nx, v, dt)

    mat_real = np.zeros((Ns, Nx))
    mat_god = np.zeros((Ns, Nx))
    mat_cd = np.zeros((Ns, Nx))
    mat_lw = np.zeros((Ns, Nx))
    mat_fft = np.zeros((Ns, Nx))


    for i in range(Ns):
        tmp_ic = ic_mat[i, ...]
        res_god = evo(T, dt, M_god, tmp_ic)
        res_lw = evo(T, dt, M_lw, tmp_ic)
        res_cd = evo(T, dt, M_cd, tmp_ic)
        res_fft = evo_fft(T, dt, v, tmp_ic)
        real = get_real(tmp_ic, int(v * T / Lx * Nx))

        mat_real[i, ...] = real
        mat_god[i, ...] = res_god
        mat_lw[i, ...] = res_lw
        mat_cd[i, ...] = res_cd
        mat_fft[i, ...] = res_fft


    with open(Data_name, 'wb') as ss:
        np.save(ss, mat_real)
        np.save(ss, mat_god)
        np.save(ss, mat_lw)
        np.save(ss, mat_cd)
        np.save(ss, mat_fft)

    tt = ic_mat[0, ...]
    res_god = evo(T, dt, M_god, tt)
    res_lw = evo(T, dt, M_lw, tt)
    res_cd = evo(T, dt, M_cd, tt)
    res_fft = evo_fft(T, dt, v, tt)

    real = get_real(tt, int(v*T/Lx*Nx))
    plt.plot(points, tt, 'r')
    plt.plot(points, res_lw, 'b')
    plt.plot(points, res_cd, 'k')
    plt.plot(points, res_god, 'g')
    plt.plot(points, res_fft, 'y')
    plt.plot(points, real, 'c')
    plt.show()




