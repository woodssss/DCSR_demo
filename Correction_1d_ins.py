import matplotlib.pyplot as plt
import numpy as np

filename = 'results_1d_spectral_god.npy'
with open(filename, 'rb') as ss:
    test_god = np.load(ss)
    real = np.load(ss)
    bpd_god = np.load(ss)
    ipd_god = np.load(ss)

filename = 'results_1d_spectral_fft.npy'
with open(filename, 'rb') as ss:
    test_fft = np.load(ss)
    real = np.load(ss)
    bpd_fft = np.load(ss)
    ipd_fft = np.load(ss)

filename = 'results_1d_spectral_lw.npy'
with open(filename, 'rb') as ss:
    test_lw = np.load(ss)
    real = np.load(ss)
    bpd_lw = np.load(ss)
    ipd_lw = np.load(ss)


filename = 'results_1d_spectral_white.npy'
with open(filename, 'rb') as ss:
    test_white = np.load(ss)
    real = np.load(ss)
    bpd_white = np.load(ss)
    ipd_white = np.load(ss)

filename = 'results_1d_spectral_pink.npy'
with open(filename, 'rb') as ss:
    test_pink = np.load(ss)
    real = np.load(ss)
    bpd_pink = np.load(ss)
    ipd_pink = np.load(ss)

filename = 'results_1d_spectral_brown.npy'
with open(filename, 'rb') as ss:
    test_brown = np.load(ss)
    real = np.load(ss)
    bpd_brown = np.load(ss)
    ipd_brown = np.load(ss)


from config.config_1d import *

bs_plt = 5
nrows = 6
fig1, ax = plt.subplots(nrows, bs_plt, figsize=(bs_plt * 4, nrows*3))
for i in range(1, bs_plt):
    ax[0, i].plot(points, test_fft[i+1, ...], 'r')
    ax[0, i].plot(points, ipd_fft[i+1, ..., 0], 'b', linewidth=2)
    ax[0, i].plot(points, real[i+1, ...], 'c--', linewidth=2)

    ax[1, i].plot(points, test_god[i+1, ...], 'r')
    ax[1, i].plot(points, ipd_god[i+1, ..., 0], 'b', linewidth=2)
    ax[1, i].plot(points, real[i+1, ...], 'c--', linewidth=2)

    ax[2, i].plot(points, test_lw[i+1, ...], 'r')
    ax[2, i].plot(points, ipd_lw[i+1, ..., 0], 'b', linewidth=2)
    ax[2, i].plot(points, real[i+1, ...], 'c--', linewidth=2)

    ax[3, i].plot(points, test_white[i + 1, ...], 'r')
    ax[3, i].plot(points, ipd_white[i + 1, ..., 0], 'b', linewidth=2)
    ax[3, i].plot(points, real[i + 1, ...], 'c--', linewidth=2)

    ax[4, i].plot(points, test_pink[i + 1, ...], 'r')
    ax[4, i].plot(points, ipd_pink[i + 1, ..., 0], 'b', linewidth=2)
    ax[4, i].plot(points, real[i + 1, ...], 'c--', linewidth=2)

    ax[5, i].plot(points, test_brown[i + 1, ...], 'r')
    ax[5, i].plot(points, ipd_brown[i + 1, ..., 0], 'b', linewidth=2)
    ax[5, i].plot(points, real[i + 1, ...], 'c--', linewidth=2)

ax[0, 0].text(0.5, 0.5, 'FFT', fontsize=36, ha='center', va='center')
ax[1, 0].text(0.5, 0.5, 'God', fontsize=36, ha='center', va='center')
ax[2, 0].text(0.5, 0.5, 'LW', fontsize=36, ha='center', va='center')
ax[3, 0].text(0.5, 0.5, 'White', fontsize=36, ha='center', va='center')
ax[4, 0].text(0.5, 0.5, 'Pink', fontsize=36, ha='center', va='center')
ax[5, 0].text(0.5, 0.5, 'Brown', fontsize=36, ha='center', va='center')

for axs in ax.flat:
    axs.set_xticks([])
    axs.set_yticks([])
    axs.axis('off')

fig1.legend(
        labels=['LF', 'IPD', 'HF'],  # Labels for the legend
        loc='upper center',  # Position the legend at the top
        bbox_to_anchor=(0.5, 0.95),  # Fine-tune the position
        fontsize=36,
        ncol=3  # Number of columns
    )
fig1.tight_layout(rect=[0, 0, 1, 0.85])
plt.savefig('1d_ins.png', bbox_inches='tight')


bs_plt = 5
nrows = 2
fig2, ax = plt.subplots(nrows, bs_plt, figsize=(bs_plt * 4, nrows*3))
for i in range(1, bs_plt):
    j=i
    ax[0, i].plot(points, test_white[j+1, ...], 'r')
    ax[0, i].plot(points, bpd_white[j+1, ..., 0], 'y', linewidth=2)
    ax[0, i].plot(points, ipd_white[j + 1, ..., 0], 'b', linewidth=2)
    ax[0, i].plot(points, real[j + 1, ...], 'c--', linewidth=2)

    ax[1, i].plot(points, test_pink[j + 1, ...], 'r')
    ax[1, i].plot(points, bpd_pink[j + 1, ..., 0], 'y', linewidth=2)
    ax[1, i].plot(points, ipd_pink[j + 1, ..., 0], 'b', linewidth=2)
    ax[1, i].plot(points, real[j + 1, ...], 'c--', linewidth=2)

    # ax[1, i].plot(points, test_god[i+1, ...], 'r')
    # ax[1, i].plot(points, ipd_god[i+1, ..., 0], 'b', linewidth=2)
    # ax[1, i].plot(points, real[i+1, ...], 'c--', linewidth=2)
    #
    # ax[2, i].plot(points, test_pink[i + 1, ...], 'r')
    # ax[2, i].plot(points, bpd_pink[i + 1, ..., 0], 'b', linewidth=2)
    # ax[2, i].plot(points, real[i + 1, ...], 'c--', linewidth=2)
    #
    # ax[3, i].plot(points, test_pink[i + 1, ...], 'r')
    # ax[3, i].plot(points, ipd_pink[i + 1, ..., 0], 'b', linewidth=2)
    # ax[3, i].plot(points, real[i + 1, ...], 'c--', linewidth=2)

ax[0, 0].text(0.5, 0.5, 'White', fontsize=32, ha='center', va='center')
ax[1, 0].text(0.5, 0.5, 'Pink', fontsize=32, ha='center', va='center')
# ax[2, 0].text(0.5, 0.5, 'BPD\nPink', fontsize=36, ha='center', va='center')
# ax[3, 0].text(0.5, 0.5, 'IPD\nPink', fontsize=36, ha='center', va='center')

for axs in ax.flat:
    axs.set_xticks([])
    axs.set_yticks([])
    axs.axis('off')

fig2.legend(
        labels=['LF', 'BPD', 'IPD', 'HF'],  # Labels for the legend
        loc='upper center',  # Position the legend at the top
        bbox_to_anchor=(0.5, 0.95),  # Fine-tune the position
        fontsize=24,
        ncol=4  # Number of columns
    )
fig2.tight_layout(rect=[0, 0, 1, 0.85])
plt.savefig('bpd_ipd_compare.png', bbox_inches='tight')
plt.show()