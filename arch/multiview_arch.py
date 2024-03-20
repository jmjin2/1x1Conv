import torch
from torch import nn as nn
from .basicvsr_arch import BasicVSR
from utils.util import tensor2img
import cv2
import os

class MultiViewSR(nn.Module):

    def __init__(self, num_feat=64, num_block=15, load_path=None, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat
        # SR
        self.basicvsr = BasicVSR(num_feat, num_block, spynet_path)
        if load_path:
            self.basicvsr.load_state_dict(torch.load(load_path)['params'], strict=False)
        # 1x1 conv
        self.conv1x1 = nn.Conv2d(9, 3, 1, 1, 0)
        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.imgname = 0

    def forward(self, x1, x2, x3):
        b, n, c, h, w = x1.size()
        out_l = []
        view1 = self.basicvsr(x1)
        view2 = self.basicvsr(x2)
        view3 = self.basicvsr(x3)
        
        view_cat = torch.cat((view1, view2, view3), 2)
        for i in range(0, n):
            x_i = view_cat[:, i, :, :, :]
            out = self.lrelu(self.conv1x1(x_i))
            out_l.extend(out)
        output = torch.stack(out_l, dim=0)
        output.unsqueeze(0)

        imgOutput = tensor2img(output)
        save_path = './results/train/multiview'
        cv2.imwrite(os.path.join(save_path, f'{str(self.imgname)}_output.png'), imgOutput)
        self.imgname += 1

        return output
