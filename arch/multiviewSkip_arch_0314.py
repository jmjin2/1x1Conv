import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicvsr_arch import BasicVSR

class MultiViewSkipSR(nn.Module):

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
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2, x3):
        b, n, c, h, w = x1.size()
        identity = x2
        identity = torch.squeeze(identity)
        identity = F.interpolate(identity, scale_factor=4, mode='bicubic')
        identity = torch.unsqueeze(identity, 0)
        out_l = []
        view1 = self.basicvsr(x1)
        view2 = self.basicvsr(x2)
        view3 = self.basicvsr(x3)

        view_cat = torch.cat((view1, view2, view3), 2)
        for i in range(0, n):
            x_i = view_cat[:, i, :, :, :]
            out = self.relu(self.conv1x1(x_i))
            out += identity[:, i, :, :, :]
            out = self.relu(out)
            out_l.extend(out)
        output = torch.stack(out_l, dim=0)
        output = torch.unsqueeze(output, 0)

        return output
