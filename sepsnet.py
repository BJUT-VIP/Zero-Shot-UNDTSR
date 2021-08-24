import torch
from configs import Config
import torch.nn as nn



class SELayer(nn.Module):
    def __init__(self, channel, out_channel, reduction):
        super(SELayer, self).__init__()
        self.keep_x_size = nn.Conv2d(channel, out_channel, 1, bias=False)
        self.convdown = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convup = nn.Sequential(nn.Conv2d(channel // reduction, out_channel, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x1 = self.keep_x_size(x)
        y = self.convdown(x)
        y = self.avg_pool(y)
        y = self.convup(y)
        return x1*y