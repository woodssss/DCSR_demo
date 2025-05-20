from utils_2d.Net import *
import numpy as np

torch.manual_seed(25)
np.random.seed(25)

############# UNet coarse at low level 32x32 ############################
c0 = 64
gn = 4
in_cn = 1
embed_dim = 128

W_np = np.random.randn(embed_dim // 2) * 30
W = nn.Parameter(torch.from_numpy(W_np).float(), requires_grad=False).to(device)

Down_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (in_cn, c0, 2, 1, gn, embed_dim),
    (c0, 2*c0, 3, 2, gn, embed_dim),
    (2*c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0, 8 * c0, 3, 2, gn, embed_dim),
]

Mid_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
]
Up_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0 + 4*c0, 2 * c0, 3, 2, gn, embed_dim),
    (2 * c0 + 2*c0, c0, 3, 2, gn, embed_dim),
    (c0 + c0, 1, 2, 1, gn, embed_dim),
]

model_sp_gen = UNet(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_cv1_gen = UNet(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_cv2_gen = UNet(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_gen = UNet(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)

save_name_sp_gen = 'ELAS0_sp_unet_gen' + '_c0_' + str(c0)
save_name_cv1_gen = 'ELAS0_cv1_unet_gen' + '_c0_' + str(c0)
save_name_cv2_gen = 'ELAS0_cv2_unet_gen' + '_c0_' + str(c0)

############# UNet SR3 32x32->64x64 ############################
c0 = 64
gn = 4
in_cn = 2
out_cn = 1
embed_dim = 128

W_np = np.random.randn(embed_dim // 2) * 30
W = nn.Parameter(torch.from_numpy(W_np).float(), requires_grad=False).to(device)


Down_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (in_cn, c0, 2, 1, gn, embed_dim),
    (c0, 2*c0, 3, 2, gn, embed_dim),
    (2*c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0, 8 * c0, 3, 2, gn, embed_dim),
]

Mid_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
]

Up_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0 + 4*c0, 2 * c0, 3, 2, gn, embed_dim),
    (2 * c0 + 2*c0, c0, 3, 2, gn, embed_dim),
    (c0 + c0, out_cn, 2, 1, gn, embed_dim),
]

model_up_sp_0 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_cv1_0 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_cv2_0 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_0 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)

save_name_up_sp_0 = 'ELAS0_up_sp_0_unet' + '_c0_' + str(c0)
save_name_up_cv1_0 = 'ELAS0_up_cv1_0_unet' + '_c0_' + str(c0)
save_name_up_cv2_0 = 'ELAS0_up_cv2_0_unet' + '_c0_' + str(c0)
############# UNet SR3 64x64->128x128 ############################
c0 = 128
gn = 4
in_cn = 2
out_cn = 1
embed_dim = 128

W_np = np.random.randn(embed_dim // 2) * 30
W = nn.Parameter(torch.from_numpy(W_np).float(), requires_grad=False).to(device)


Down_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (in_cn, c0, 2, 1, gn, embed_dim),
    (c0, 2*c0, 3, 2, gn, embed_dim),
    (2*c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0, 8 * c0, 3, 2, gn, embed_dim),
]

Mid_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
]

Up_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0 + 4*c0, 2 * c0, 3, 2, gn, embed_dim),
    (2 * c0 + 2*c0, c0, 3, 2, gn, embed_dim),
    (c0 + c0, out_cn, 2, 1, gn, embed_dim),
]

model_up_sp_1 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_cv1_1 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_cv2_1 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_1 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)


save_name_up_sp_1 = 'ELAS0_up_sp_1_unet' + '_c0_' + str(c0)
save_name_up_cv1_1 = 'ELAS0_up_cv1_1_unet' + '_c0_' + str(c0)
save_name_up_cv2_1 = 'ELAS0_up_cv2_1_unet' + '_c0_' + str(c0)
############# UNet SR3 128x128 -> 256x256 ############################
c0 = 256
gn = 4
in_cn = 2
out_cn = 1
embed_dim = 128

W_np = np.random.randn(embed_dim // 2) * 30
W = nn.Parameter(torch.from_numpy(W_np).float(), requires_grad=False).to(device)


Down_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (in_cn, c0, 2, 1, gn, embed_dim),
    (c0, 2*c0, 3, 2, gn, embed_dim),
    (2*c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0, 8 * c0, 3, 2, gn, embed_dim),
]

Mid_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
    (8 * c0, 8 * c0, 1, 1, gn, embed_dim),
]

Up_config = [
    # Tuple: (in_cn, out_cn, kernel_size, stride, embed_dim, n_head, head_dim)
    (8 * c0, 4 * c0, 3, 2, gn, embed_dim),
    (4 * c0 + 4*c0, 2 * c0, 3, 2, gn, embed_dim),
    (2 * c0 + 2*c0, c0, 3, 2, gn, embed_dim),
    (c0 + c0, out_cn, 2, 1, gn, embed_dim),
]

model_up_sp_2 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_cv1_2 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_cv2_2 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
model_up_2 = UNet_cond(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)

save_name_up_sp_2 = 'ELAS0_up_sp_2_unet' + '_c0_' + str(c0)
save_name_up_cv1_2 = 'ELAS0_up_cv1_2_unet' + '_c0_' + str(c0)
save_name_up_cv2_2 = 'ELAS0_up_cv2_2_unet' + '_c0_' + str(c0)
###################################################################################