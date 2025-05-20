import numpy as np
import matplotlib.pyplot as plt
import sys

def kernal(xs, ys, l):
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    return np.exp(-((np.sin(np.pi * dx) / l) ** 2) / 2)


def generate_grf(Ns, points, l):
    Nx = points.shape[0]
    Corv = kernal(points, points, l)
    g_mat = np.zeros((Ns, Nx))
    mean = np.zeros_like(points)

    for i in range(Ns):
        tmp = np.random.multivariate_normal(mean, Corv)
        g_mat[[i], :] = tmp/np.max(np.abs(tmp))

    return g_mat

def generate_sharp_grf(Ns, points, l):
    Nx = points.shape[0]
    Corv = kernal(points, points, l)
    g_mat = np.zeros((Ns, Nx))
    mean = np.zeros_like(points)

    for i in range(Ns):
        tmp = np.random.multivariate_normal(mean, Corv)
        tmp /= np.max(np.abs(tmp))

        tmp = np.where(tmp > 0, 1, 0)
        tmp[points<0.2] = 0
        tmp[points>0.8] = 0
        g_mat[[i], :] = tmp

    return g_mat


