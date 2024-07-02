import torch
from torch import nn as nn
from .basicvsr_arch import BasicVSR
import cv2
import os

class FusionA(nn.Module):
    def __init__(self, num_feat=64, num_block=15, res_depth=6, load_path=None, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat
        # SR
        self.basicvsr = BasicVSR(num_feat, num_block, spynet_path)
        if load_path:
            self.basicvsr.load_state_dict(torch.load(load_path)['params'], strict=False)

        self.conv_first = nn.Conv2d(9, 64, 3, 1, 1)

        # Attention module


        # feature extract
        self.residual = self.make_layer(ResBlock, res_depth)

        # 1x1 conv
        self.conv1x1 = nn.Conv2d(64, 3, 1, 1, 0)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3):
        b, n, c, h, w = x1.size()
        out_l = []
        view1 = self.basicvsr(x1)
        view2 = self.basicvsr(x2)
        view3 = self.basicvsr(x3)
        # view.shape: [1, 15, 3, 256, 256]
        view_cat = torch.cat((view1, view2, view3), 2)
        # view_cat.shape: [1, 15, 9, 256, 256]
        for i in range(0, n):
            x_i = view_cat[:, i, :, :, :]
            out = self.lrelu(self.conv_first(x_i))
            out = self.lrelu(self.residual(out))
            out = self.lrelu(self.conv1x1(out))
            out_l.extend(out)
        output = torch.stack(out_l, dim=0)
        output = torch.unsqueeze(output, 0)

        return output

class ResBlock(nn.Module):    
    def __init__(self):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 
    
