# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com
import time

from src.models.modules import *

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NetG(nn.Module):
    def __init__(self, ngf, nz, dim, imsize, ch_size, clip_type="ViT-B/32"):
        super(NetG, self).__init__()
        self.ngf = ngf
        # input noise (batch_size, 100)

        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 4 * 4),
            nn.ReLU()
        )

        self.c_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # build GBlocks
        self.GBlocks = nn.ModuleList([])

        self.GBlocks.append(G_Block(dim + nz, 256, 256, upsample=True))
        self.GBlocks.append(G_Block(dim + nz, 256, 256, upsample=True))
        self.GBlocks.append(G_Block(dim + nz, 256, 256, upsample=True))
        self.GBlocks.append(G_Block(dim + nz, 256, 128, upsample=True))
        self.GBlocks.append(G_Block(dim + nz, 128, 64, upsample=True))
        self.GBlocks.append(G_Block(dim + nz, 64, 32, upsample=True))

        # self.ks = KSBLK(dim + nz, out_ch)
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, ch_size, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, c):  # x=noise, c=ent_emb
        c = c.float()

        c = self.c_fc(c)
        # concat noise and sentence
        # (bs, 4096)
        out = self.fc(noise)
        # (bs, 256, 4, 4)
        out1 = out.view(noise.size(0), 8*self.ngf, 4, 4)

        cond = torch.cat((noise, c), dim=1)

        for i in range(len(self.GBlocks)):
            gblock = self.GBlocks[i]
            out1 = gblock(out1, cond)
        # convert to RGB image
        # (bs, 3, 256, 256)
        out = self.to_rgb(out1)
        return out


class SemanticCorrection(nn.Module):
    def __init__(self, clip_model, dim, ch_size):
        super(SemanticCorrection, self).__init__()
        self.img_encoder = CLIP_Image_Encoder(clip_model)

        self.activate = nn.Softmax(-1)

        self.linear = nn.Linear(50, dim)

        self.conv = nn.Conv2d(3, 32, 3, 1, 1)

        self.GBlocks2 = nn.ModuleList(
            [G_Block(dim, 3, 32, upsample=False),
             G_Block(dim, 32, 64, upsample=False),
             G_Block(dim, 64, 32, upsample=False)]
        )

        # self.ks = KSBLK(dim + nz, out_ch)
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, ch_size, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, out, cond):
        # start = time.time()
        output_embed, output_feat = self.img_encoder(out)

        # end1 = time.time()
        cond2 = torch.cosine_similarity(output_feat, cond)
        # end2 = time.time()
        # cond2 = cond2.squeeze(-1)
        # cond2 = self.activate(cond2)
        # cond2 = self.linear(cond2)

        new_cond = cond2.unsqueeze(1) * cond
        img_feat = self.conv(out)
        # end3 = time.time()
        for i in range(len(self.GBlocks2)):
            gblock = self.GBlocks2[i]
            out = gblock(out, new_cond)
        end4 = time.time()
        out2 = out + img_feat
        # convert to RGB image
        # (bs, 3, 256, 256)
        out2 = self.to_rgb(out2)
        # end5 = time.time()

        # print(end1 - start, end2 - end1, end3 - end2, end4 - end3, end5 - end4)
        return out2

    def cosine_similarity_onnx_exportable(self, x1, x2, dim=-1):
        cross = (x1 * x2).sum(dim=dim)
        x1_l2 = (x1 * x1).sum(dim=dim)
        x2_l2 = (x2 * x2).sum(dim=dim)
        return torch.div(cross, (x1_l2 * x2_l2).sqrt())


class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        # x:
        if self.learnable_sc:
            #
            x = self.c_sc(x)
        return x

    def residual(self, h, w):
        # h: (bs, 256, 8)  y:(1, 868)
        # (bs, 256, 8, 8)
        h1 = self.fuse1(h, w)
        # (bs, 256, 8, 8)
        h1 = self.c1(h1)
        # (bs, 256, 8, 8)
        h1 = self.fuse2(h1, w)
        # (bs, 256, 8, 8)
        h1 = self.c2(h1)
        return h1

    def forward(self, x, y):
        # x: (bs, 256, 4, 4) y:(1, 868)
        if self.upsample == True:
            # (bs, 256, 8, 8)
            x = F.interpolate(x, scale_factor=2)
        out = self.shortcut(x) + self.residual(x, y)
        return out


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize)) - 1
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs, channel_nums

