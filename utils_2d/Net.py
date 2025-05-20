import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DownBlock(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False)
        if out_cn > 32:
            self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
        else:
            self.groupnorm = nn.GroupNorm(num_groups=4, num_channels=out_cn)
        #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
        self.act = lambda x: x * torch.sigmoid(x)

        #self.conv2 = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False)

    def forward(self, x):
        h = self.conv(x)
        h = self.groupnorm(h)
        return self.act(h)

class UpBlock(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn):
        super(UpBlock, self).__init__()
        self.out_cn = out_cn
        if out_cn >= 32:
            self.conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride, bias=False)
            self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
        else:
            self.conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride)
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self, x):
        h = self.conv(x)
        if self.out_cn >= 32:
            h = self.groupnorm(h)
            return self.act(h)
        else:
            #return self.act(h + self.dense(embed))
            return h

class MidBlock(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn):
        super(MidBlock, self).__init__()
        self.conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride)
        self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=in_cn, eps=1e-6, affine=True)
        #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self, x):
        h = self.conv(x)
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
        return self.dense(x)[..., None, None]

class DownBlock_t(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(DownBlock_t, self).__init__()
        self.conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False)
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
            self.conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride, bias=False)
            self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
            #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
            self.dense = Dense(embed_dim, out_cn)
        else:
            self.conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride)
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
        self.conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride)
        self.dense = Dense(embed_dim, out_cn)
        self.groupnorm = nn.GroupNorm(num_groups=gn, num_channels=in_cn, eps=1e-6, affine=True)
        #self.groupnorm = nn.BatchNorm1d(num_features=out_cn)
        self.act = lambda x: x * torch.sigmoid(x)
    def forward(self, x, embed):
        h = self.conv(x)
        h += self.dense(embed)
        h = self.groupnorm(h)
        return self.act(h)


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
        # x [bs, Nx, Nx, 1]
        embed = self.act(self.embed(t/self.Nt))

        h = x.permute(0, 3, 1, 2)

        # print(x.shape, h.shape)
        # zxc

        h_ls = [h]

        for layer in self.Down_layers:
            h = layer(h, embed)
            h_ls.append(h)
            #print(h.shape)

        for layer in self.Mid_layers:
            #print('mm', h.shape)
            h = layer(h, embed)

        h_ls.pop()
        h = self.Up_layers[0](h, embed)

        for layer in self.Up_layers[1:]:
            bh = h_ls.pop()
            #print('uu', h.shape, bh.shape)
            h = layer(torch.cat([h, bh], dim=1), embed)


        out = h.permute(0, 2, 3, 1)
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



class UNet_cond(nn.Module):
    def __init__(self, Nt, embed_dim, W, Down_config, Mid_config, Up_config):
        super(UNet_cond, self).__init__()
        self.act = lambda x: x * torch.sigmoid(x)
        self.embed = nn.Sequential(GaussianFourierProjection(W),
                                   nn.Linear(embed_dim, embed_dim))

        self.Down_layers = self._create_down_layer(Down_config)
        self.Mid_layers = self._create_mid_layer(Mid_config)
        self.Up_layers = self._create_up_layer(Up_config)

        self.Nt = Nt

    def forward(self, x, a, t):
        # x [bs, Nx, Nx, 3]
        # a [bs, Nx, Nx, 1]
        embed = self.act(self.embed(t/self.Nt))

        x = torch.cat([x, a], dim=-1)

        h = x.permute(0, 3, 1, 2)

        #print(h.shape)

        h_ls = [h]

        for layer in self.Down_layers:
            h = layer(h, embed)
            h_ls.append(h)
            #print(h.shape)

        for layer in self.Mid_layers:
            #print('mm', h.shape)
            h = layer(h, embed)

        h_ls.pop()
        h = self.Up_layers[0](h, embed)

        for layer in self.Up_layers[1:]:
            bh = h_ls.pop()
            #print('uu', h.shape, bh.shape)
            h = layer(torch.cat([h, bh], dim=1), embed)

        h = h.permute(0, 2, 3, 1)

        # h = self.fc1(h)
        # h = F.relu(h)
        # h = self.fc2(h)
        return h

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



class Attention(nn.Module):
    def __init__(self, dim, n_head=4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = n_head
        hidden_dim = dim_head * n_head

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class Block(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn):
        super(Block, self).__init__()
        if in_cn<=out_cn:
            self.proj = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False)
        else:
            self.proj = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride, bias=False)
        if out_cn%16==0:
            self.norm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
        else:
            self.norm = nn.GroupNorm(num_groups=out_cn, num_channels=out_cn)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(ResBlock, self).__init__()
        self.block1 = Block(in_cn, out_cn, kernel_size, stride, gn)
        self.block2 = Block(out_cn, out_cn, kernel_size=1, stride=1, gn=gn)

        if in_cn<=out_cn:
            self.res_conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False) if in_cn != out_cn else nn.Identity()
        else:
            self.res_conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride, bias=False) if in_cn != out_cn else nn.Identity()
        self.dense = Dense(embed_dim, out_cn)


    def forward(self, x, embed):
        h = self.block1(x)
        ##### check here ################
        #print(h.shape, (self.dense(embed)+h).shape)
        h = h + self.dense(embed)
        h = self.block2(h)
        return h + self.res_conv(x)

class ResBlock_attn(nn.Module):
    def __init__(self, in_cn, out_cn, kernel_size, stride, gn, embed_dim):
        super(ResBlock_attn, self).__init__()
        self.block1 = Block(in_cn, out_cn, kernel_size, stride, gn)
        self.block2 = Block(out_cn, out_cn, kernel_size=1, stride=1, gn=gn)

        if in_cn<=out_cn:
            self.res_conv = nn.Conv2d(in_cn, out_cn, kernel_size, stride, bias=False) if in_cn != out_cn else nn.Identity()
        else:
            self.res_conv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size, stride, bias=False) if in_cn != out_cn else nn.Identity()
        self.dense = Dense(embed_dim, out_cn)

        if out_cn%16==0:
            self.norm = nn.GroupNorm(num_groups=gn, num_channels=out_cn)
        else:
            self.norm = nn.GroupNorm(num_groups=out_cn, num_channels=out_cn)

        self.attn = Attention(out_cn)


    def forward(self, x, embed):
        h = self.block1(x)
        ##### check here ################
        #print(h.shape, (self.dense(embed)+h).shape)
        h = h + self.dense(embed)
        h = self.block2(h)
        h = h + self.res_conv(x)
        h = self.norm(h) if self.norm else h
        h = h + self.attn(h)
        return h

class UNet_attn_cond(nn.Module):
    def __init__(self, Nt, embed_dim, W, Down_config, Mid_config, Up_config):
        super(UNet_attn_cond, self).__init__()
        self.act = lambda x: x * torch.sigmoid(x)
        self.embed = nn.Sequential(GaussianFourierProjection(W),
                                   nn.Linear(embed_dim, embed_dim))

        self.len_mid = len(Mid_config)

        self.Down_layers = self._create_down_layer(Down_config)
        if self.len_mid > 0:
            self.Mid_layers = self._create_mid_layer(Mid_config)
        self.Up_layers = self._create_up_layer(Up_config)

        self.Nt = Nt

    def forward(self, x, a, t):
        # x [bs, Nx, Nx, 3]
        # a [bs, Nx, Nx, 1]
        embed = self.act(self.embed(t/self.Nt))

        x = torch.cat([x, a], dim=-1)
        h = x.permute(0, 3, 1, 2)

        h_ls = [h]

        for layer in self.Down_layers:
            h = layer(h, embed)
            h_ls.append(h)
            # print(h.shape)

        if self.len_mid > 0:
            for layer in self.Mid_layers:
                h = layer(h, embed)

        h_ls.pop()
        h = self.Up_layers[0](h, embed)

        for layer in self.Up_layers[1:]:
            bh = h_ls.pop()
            # print('uu', h.shape, bh.shape)
            h = layer(torch.cat([h, bh], dim=1), embed)

        out = h.permute(0, 2, 3, 1)
        return out

    def _create_up_layer(self, config):
        layers = nn.ModuleList()
        for k in config:
            if k[6] == 'attn':
                tmp_layer = ResBlock_attn(k[0], k[1], k[2], k[3], k[4], k[5])
            else:
                tmp_layer = ResBlock(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers

    def _create_down_layer(self, config):
        layers = nn.ModuleList()
        for k in config:
            if k[6] == 'attn':
                tmp_layer = ResBlock_attn(k[0], k[1], k[2], k[3], k[4], k[5])
            else:
                tmp_layer = ResBlock(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers

    def _create_mid_layer(self, config):
        layers = nn.ModuleList()
        for k in config:
            if k[6] == 'attn':
                tmp_layer = ResBlock_attn(k[0], k[1], k[2], k[3], k[4], k[5])
            else:
                tmp_layer = ResBlock(k[0], k[1], k[2], k[3], k[4], k[5])
            layers.append(tmp_layer)

        return layers