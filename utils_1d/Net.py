import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DownBlock_t(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(DownBlock_t, self).__init__()
        self.conv = nn.Conv1d(in_cn, out_cn, kernel_size, stride, bias=False)
        self.dense = Dense(embed_dim, out_cn)
        if out_cn > 32:
            self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
        else:
            self.groupnorm = nn.GroupNorm(num_groups=4, num_channels=out_cn)
        #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
        self.act = lambda x: x * torch.sigmoid(x)

        #self.conv2 = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False)

    def forward(self, x, embed):
        h = self.conv(x)
        h += self.dense(embed)
        h = self.groupnorm(h)
        return self.act(h)

class UpBlock_t(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(UpBlock_t, self).__init__()
        self.out_cn = out_cn
        if out_cn >= 32:
            self.conv = nn.ConvTranspose1d(in_cn, out_cn, kernel_size, stride, bias=False)
            self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
            #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
            self.dense = Dense(embed_dim, out_cn)
        else:
            self.conv = nn.ConvTranspose1d(in_cn, out_cn, kernel_size, stride)
            self.dense = Dense(embed_dim, out_cn)
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self, x, embed):
        h = self.conv(x)
        if self.out_cn >= 32:
            h += self.dense(embed)
            h = self.groupnorm(h)
            return self.act(h)
        else:
            #return self.act(h + self.dense(embed))
            return h

class MidBlock_t(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(MidBlock_t, self).__init__()
        self.conv = nn.Conv1d(in_cn, out_cn, kernel_size, stride)
        self.dense = Dense(embed_dim, out_cn)
        self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=in_cn, eps=1e-6, affine=True)
        #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self, x, embed):
        h = self.conv(x)
        h += self.dense(embed)
        h = self.groupnorm(h)
        return self.act(h)

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, W):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        #self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False).to(device)
        self.W = W

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, x):
        return self.dense(x)[..., None]


class UNet(nn.Module):
    def __init__(self, Nt, embed_dim, W, Down_config, Mid_config, Up_config):
        super(UNet, self).__init__()
        self.act = lambda x: x * torch.sigmoid(x)
        self.embed = nn.Sequential(GaussianFourierProjection(W),
                                   nn.Linear(embed_dim, embed_dim))

        self.Down_layers = self._create_down_layer(Down_config)
        self.Mid_layers = self._create_mid_layer(Mid_config)
        self.Up_layers = self._create_up_layer(Up_config)

        self.Nt = Nt

    def forward(self, x, t):
        # x [bs, Nx, 1]
        embed = self.act(self.embed(t/self.Nt))

        h = x.permute(0, 2, 1)

        h_ls = [h]

        for layer in self.Down_layers:
            h = layer(h, embed)
            h_ls.append(h)

        for layer in self.Mid_layers:
            #print('mm', h.shape)
            h = layer(h, embed)

        h_ls.pop()
        h = self.Up_layers[0](h, embed)

        for layer in self.Up_layers[1:]:
            bh = h_ls.pop()
            #print('uu', h.shape, bh.shape)
            h = layer(torch.cat([h, bh], dim=1), embed)


        out = h.permute(0, 2, 1)
        return out

    def _create_up_layer(self, config):
        layers = nn.ModuleList()
        #print(config)
        for k in config:
            #print(k[0], k[1], k[2], k[3], k[4], k[5])
            tmp_layer = UpBlock_t(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers

    def _create_down_layer(self, config):
        layers = nn.ModuleList()
        for k in config:
            tmp_layer = DownBlock_t(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers

    def _create_mid_layer(self, config):
        layers = nn.ModuleList()
        for k in config:
            tmp_layer = MidBlock_t(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers