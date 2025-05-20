from utils_1d.Net import *
import numpy as np

torch.manual_seed(25)
np.random.seed(25)
############# UNet up 16x16->32x32 ############################
c0 = 64
gn = 4
in_cn = 1
out_cn = 1
embed_dim = 64
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

model_c_gen = UNet(1, embed_dim, W, Down_config, Mid_config, Up_config).to(device)
save_name_c_gen = 'SC_1d_unet_gen' + '_c0_' + str(c0)