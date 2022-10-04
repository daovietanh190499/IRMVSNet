import torch
import torch.nn as nn
from itertools import product
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data
import torch.nn.functional as NF

from unet_ds2gn import UNetDS2GN
from homo_warping import *

from os import path
import gc
import sys
import time

class GRU(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super(GRU, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = torch.sigmoid(rn)
        uns = torch.sigmoid(un)
        return rns, uns

    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h = None):
        N, C, H, W = x.shape
        HC = self.output_channel
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
        return u * h + (1 - u) * y
    
class IRMVSNet(nn.Module):
    def __init__(self, size, device):
        super(IRMVSNet, self).__init__()
        # setup network modules
        
        if size is not None and device is not None:
            self.pixel_coords = init_pixel_coords(*size).to(device)
            self.down_pixel_coords = init_pixel_coords(int(size[0]/4), int(size[1]/4)).to(device)

        self.feature_extractor = UNetDS2GN()

        gru_input_size = self.feature_extractor.output_size
        gru1_output_size = 16
        gru2_output_size = 4
        gru3_output_size = 2
        self.gru1 = GRU(gru_input_size, gru1_output_size, 3)
        self.gru2 = GRU(gru1_output_size, gru2_output_size, 3)
        self.gru3 = GRU(gru2_output_size, gru3_output_size, 3)

        self.prob_conv = nn.Conv2d(2, 1, 3, 1, 1)
    
    def depth_compute(self, prob_volume, depth_values):
        _, M, H, W = prob_volume.shape
        softmax_probs = torch.softmax(prob_volume, 1)
        probs, indices = softmax_probs.max(1)
        depths = torch.index_select(depth_values, 0, indices.flatten())
        depth_image = depths.view(H, W)
        prob_image = probs.view(H, W)
        return depth_image, prob_image

    def forward(self, images, Ks, Rts, depth_values):
        N, C, IH, IW = images.shape
        Hos = get_homographies(Ks, Rts, depth_values, 1)
        Hs = get_homographies(Ks, Rts, depth_values, 4) # nViews x nDepths x 3 x 3
        
        if self.pixel_coords is None:
            self.pixel_coords = get_pixel_coords(images[0].unsqueeze(0)).to(images.device)
        
        f = []
        for i in range(N):
            img = images[i].unsqueeze(0)
            if i > 0:
                img = homography_transform(img, Hos[i-1, -1, :, :].unsqueeze(0), self.pixel_coords)
            fea = self.feature_extractor(img)
            f.append(fea)
        
        if self.down_pixel_coords is None:
            self.down_pixel_coords = get_pixel_coords(f[0]).to(f[0].device)

        cost_1 = None
        cost_2 = None
        cost_3 = None
        depth_costs = []
        
        for i in range(Hs.size()[1]):
            warped = f[0]
            for j in range(Hs.size()[0]):
                fea_ = homography_transform(f[j+1], torch.matmul(torch.inverse(Hs[j, -1]), Hs[j, i]).unsqueeze(0), self.down_pixel_coords)
                warped = torch.cat((warped, fea_), 0)
            
            cost_d = (warped ** 2).mean(0) - (warped.mean(0))**2
            cost_d = cost_d.unsqueeze(0)
            cost_1 = self.gru1(-cost_d, cost_1)
            cost_2 = self.gru2(cost_1, cost_2)
            cost_3 = self.gru3(cost_2, cost_3)
            
            reg_cost = self.prob_conv(cost_3) # 1 x 1 x H x W
            depth_costs.append(reg_cost) # D of 1 x 1 x H x W
        
        prob_volume = torch.cat(depth_costs, 1) # 1 x D x H x W
        
        depth_img, prob_img = self.depth_compute(prob_volume, depth_values) # H x W and H x W
        
        return prob_volume, depth_img, prob_img


def irmvsnet_loss(prob_volume, depth_gt, depth_values):
    depth_gt = F.interpolate(depth_gt.unsqueeze(0).unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=True, recompute_scale_factor=True).squeeze(0).squeeze(0)
    mask = depth_gt != 0
    depth_value_mat = depth_values.repeat(depth_gt.shape[0], depth_gt.shape[1], 1).permute(2,0,1)
    gt_index_image = torch.argmin(torch.abs(depth_value_mat.unsqueeze(0)-depth_gt.unsqueeze(0).unsqueeze(0)), dim=1).squeeze(0).squeeze(0)
    gt_index_image = gt_index_image*mask + ~mask*(-100)
    loss = F.cross_entropy(prob_volume, gt_index_image.unsqueeze(0), ignore_index=-100, reduction='mean')
    return loss