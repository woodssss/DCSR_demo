import numpy as np
from scipy.sparse import diags

def upwind_matrix(Nx, dx, v, dt):
    A = np.zeros((Nx, Nx))
    for i in range(Nx):
        A[i, i] = 1/dx
    for i in range(1, Nx):
        A[i, i-1] = -1/dx

    A[0, -1] = -1/dx

    M = np.eye(Nx) - dt*A*v
    return M

def central_difference_matrix(Nx, dx, v, dt):
    A = np.zeros((Nx, Nx))
    for i in range(1, Nx):
        A[i, i-1] = -1/dx/2
    for i in range(Nx-1):
        A[i, i + 1] = 1 / dx/2
    return np.eye(Nx) - dt*A*v

# Construct Lax-Friedrichs matrix
def lax_friedrichs_matrix(nx, dx, v, dt):
    alpha = v * dt / dx
    # Diagonals for the Lax-Friedrichs matrix
    main_diag = np.zeros(nx)
    upper_diag = 0.5 * (1 - alpha) * np.ones(nx - 1)
    lower_diag = 0.5 * (1 + alpha) * np.ones(nx - 1)

    # Periodic boundary conditions
    upper_diag = np.append(upper_diag, 0.5 * (1 - alpha))
    lower_diag = np.append(0.5 * (1 + alpha), lower_diag)

    # Create sparse matrix
    A = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], shape=(nx, nx)).toarray()
    return A

# Construct Lax-Wendroff matrix
def lax_wendroff_matrix(nx, dx, v, dt):
    alpha = v * dt / dx
    beta = (v * dt / dx) ** 2
    # Diagonals for the Lax-Wendroff matrix
    main_diag = (1 - beta) * np.ones(nx)
    upper_diag = (-0.5 * alpha + 0.5 * beta) * np.ones(nx - 1)
    lower_diag = (0.5 * alpha + 0.5 * beta) * np.ones(nx - 1)

    # Periodic boundary conditions
    upper_diag = np.append(-0.5 * alpha + 0.5 * beta, upper_diag)
    lower_diag = np.append(0.5 * alpha + 0.5 * beta, lower_diag)

    # Create sparse matrix
    B = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], shape=(nx, nx)).toarray()
    return B

def godunov_matrix(nx, dx, v, dt):
    alpha = v * dt / dx
    main_diag = (1 - alpha) * np.ones(nx)
    lower_diag = alpha * np.ones(nx - 1)
    # Periodic boundary condition
    lower_diag = np.append(alpha, lower_diag)
    G = diags([main_diag, lower_diag], [0, -1], shape=(nx, nx)).toarray()
    return G


def evo(T, dt, M, ic):
    u = ic.copy()

    for i in range(int(T/dt)):
        u = np.matmul(M, u)

    return u

def get_real(ic, move):
    res = np.roll(ic, move)
    return res

def evo_fft(T, dt, v, ic):
    N = ic.shape[0]
    k = np.fft.fftfreq(N, d=1/N) * 2 * np.pi
    num_steps = int(T / dt)

    u = ic.copy()

    for step in range(num_steps):
        # Compute the Fourier transform of u
        u_hat = np.fft.fft(u)

        # Update in Fourier space using the advection equation
        u_hat = u_hat * np.exp(-1j * v * k * dt)

        # Transform back to physical space
        u = np.fft.ifft(u_hat).real

    return u





