import torch.nn as nn
import torch.nn.functional as F
from base_networks import *
from configs import Config

class SELayer(nn.Module):  # 用1*1的卷积代替全连接层，并使输入通道和输出通道可以任意调整
    def __init__(self, channel, out_channel, reduction):
        super(SELayer, self).__init__()
        self.keep_x_size = nn.Conv2d(channel, out_channel, 1, bias=False)
        # self.ca_res = nn.MaxPool2d(1)
        self.convdown = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.MaxPool2d(1)
        self.convup = nn.Sequential(nn.Conv2d(channel // reduction, out_channel, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x1 = self.keep_x_size(x)
        # y_res = self.ca_res(x1)
        y = self.convdown(x)
        # y = self.avg_pool(y)
        y = self.max_pool(y)
        # y1 = torch.var(y, dim=2, keepdim=True)
        # y1 = torch.var(y1, dim=3, keepdim=True)
        # y = torch.add(y1, y2)
        y = self.convup(y)
        # y = torch.add(y, y_res)
        return x1*y

class simpleNet(nn.Module):
    def __init__(self, Y=True, scale_factor=Config.sr):
        super(simpleNet, self).__init__()
        base_filter = 64
        num_stages = 3
        d = 1
        if Y == False:
            d = 3
        if scale_factor == 2:
            kernel = 4
            stride = 2
            padding = 1
        elif scale_factor == 4:
            kernel = 4
            stride = 2
            padding = 1
            # Initial Feature Extraction

        self.feat0 = ConvBlock(d, base_filter, 1, 1, 0, activation='relu')
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = UpBlock(base_filter, kernel, stride, padding)
        # self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        # self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        # self.output = nn.Conv2d(base_filter*2, 3, 3, 1, 1)
        self.se1 = SELayer(base_filter*3, base_filter, 2)
        # self.output = nn.Conv2d(base_filter, 3, 1)
        self.se2 = SELayer(base_filter, 3, 1)
        # self.ps = nn.PixelShuffle(int(2))
        # self.se2 = SELayer(base_filter, d, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        x = self.feat0(x)  # 1, 128, 128, 128
        h1 = self.up1(x)  # 1, 128, 256, 256
        l1 = self.down1(h1)  # 1, 128, 128, 128
        concat_l1 = torch.cat((x, l1), 1)
        h2 = self.up2(concat_l1)  # 1, 128, 256, 256
        concat_h2 = torch.cat((h2, h1), 1)
        l2 = self.down2(concat_h2) # 1, 128, 128, 128
        l2 = x + l2
        concat_l2 = torch.cat((l2, concat_l1), 1)
        h3 = self.up3(concat_l2)   # 1, 128, 256, 256
        concat_h3 = torch.cat((h3, concat_h2), 1)

        out = self.se1(concat_h3)  # 1, 128, 256, 256
        h4 = self.up4(out)
        out = self.se2(h4)  # 1, 3, 256, 256
        return out