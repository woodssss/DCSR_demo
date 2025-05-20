# Diffusion-Based Correction and Super-Resolution
## This project provides code for **Improving Data Fidelity via Diffusion Model-Based Correction and Super-Resolution**


# Usage of code (1D)
## This section outlines the standard procedure for utilizing the code for 1D advection equation with bump initial conditions.

# **Step 1.** Data preparation
## ` python prepare_1d_data.py`

# **Step 2.** Training
## ` python train_1d.py`

# **Step 3.** Improve data fidelity
- ## ` python Correction_1d.py --type lw`
## Correction LF data obtained from type: Lax-Wendroff scheme (lw), Godunov scheme (god), and FFT  (fft). Or data polluted by different type of noises (white, pink and brown).
- ## Or Run `Correction_1d.ipynb` using Jupyternotebook.

# **Step 4.** Visualize results
- ## Run `Correction_1d_various.ipynb` using Jupyternotebook.


# Usage of code (2D)
## This section outlines the standard procedure for utilizing the code for 2D, with a focus on the implementation of the 2D climate dataset.  

# **Step 1.** Data preparation

  - ## Option A (recommended): you can download the prepared datasets from Google Drive by
    ## ` python download_data.py  `
    - ## The datasets are stored in the 'data' folder

  - ## Option B: prepare your own datasets.
    - ## First, generate high-fidelity high-resolution (HFHR) datasets (256x256) and low-fidelity low-resolution (LFLR) datasets (32x32), and store in the raw_data folder.
    - ## Then prepare the intermidiate datasets HFLR datasets at resolution 128x128, 64x64, 32x32 by downsampling the HFHR dataset using cubic interpolation: 
    - ## ` python prepare_data.py --type WD ` 
    - ## WD and PV are, NS stands for Navier-Stokes dataset, ELAS0, ELAS1, ELAS2 stands for the dataset for linear elasticity.
    - ## The resulting datasets are stored in the data folder.
   
# **Step 2.** Training for correction at LR level. 

- ## Option A (recommended): you can download the pretrained models from Google Drive by
    ## ` python download_chckpt.py  `
    - ## The models are stored in the 'mdls' folder
## Option B: For completeness, the following commands outline how to train models.

   - ## Train an unconditional diffuion model for the HFLR dataset.
   - ## ` python train_gen.py --type WD --smth cv1` for HFLR dataset
 

# **Step 3.** Training for cascaded SR3.
## Train three conditional diffusion models for the HFLR datasets, respectively.
   - ## ` python train_sup.py --type WD --smth cv1 --flag 0` for SR from 32x32 to 64x64
   - ## ` python train_sup.py --type WD --smth cv1 --flag 1` for SR from 64x64 to 128x128
   - ## ` python train_sup.py --type WD --smth cv1 --flag 2` for SR from 128x128 to 256x256

# **Step 4.** Improve data fidelity.
  - ## ` python DCSR_WD.py`

# **Step 5.** Visulization.
- ## Run ` Comp_WD.ipynb` for Wind Speed ERA5 dataset
- ## Run ` Comp_PV.ipynb` for Potential Velocity ERA5 dataset
- ## Run ` Comp_NS.ipynb` for Navier-Stokes dataset
- ## Run ` Comp_ELAS.ipynb` for Linear Elasticity dataset
