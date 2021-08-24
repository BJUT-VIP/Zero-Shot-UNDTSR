import torch
import torch.nn as nn

class SELayer(nn.Module):  # 用1*1的卷积代替全连接层，并使输入通道和输出通道可以任意调整
    def __init__(self, channel, out_channel, reduction):
        super(SELayer, self).__init__()
        self.keep_x_size = nn.Conv2d(channel, out_channel, 1, bias=False)
        # self.ca_res = nn.MaxPool2d(1)
        self.convdown = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.avg_pool = nn.MaxPool2d(1)
        self.convup = nn.Sequential(nn.Conv2d(channel // reduction, out_channel, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x1 = self.keep_x_size(x)
        # y_res = self.ca_res(x1)
        y = self.convdown(x)
        y = self.avg_pool(y)
        # y1 = torch.var(y, dim=2, keepdim=True)
        # y1 = torch.var(y1, dim=3, keepdim=True)
        y = self.convup(y)
        # y = torch.add(y, y_res)
        return x1*y