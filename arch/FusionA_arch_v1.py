import torch
from torch import nn as nn
from .basicvsr_arch import BasicVSR
import cv2
import os

class FusionA_v1(nn.Module):
    def __init__(self, num_feat=64, num_block=15, load_path=None, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat
        # SR
        self.basicvsr = BasicVSR(num_feat, num_block, spynet_path)
        if load_path:
            self.basicvsr.load_state_dict(torch.load(load_path)['params'], strict=False)
        # 1x1 conv
        self.conv_first = nn.Conv2d(9, 64, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(64, 3, 1, 1, 0)
        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x1, x2, x3):
        b, n, c, h, w = x1.size()
        out_l = []
        view1 = self.basicvsr(x1)
        view2 = self.basicvsr(x2)
        view3 = self.basicvsr(x3)
        
        view_cat = torch.cat((view1, view2, view3), 2)
        for i in range(0, n):
            x_i = view_cat[:, i, :, :, :]
            out = self.lrelu(self.conv_first(x_i))
            out = self.lrelu(self.conv1x1(out))
            out_l.extend(out)
        output = torch.stack(out_l, dim=0)
        output = torch.unsqueeze(output, 0)

        return output