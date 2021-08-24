import torch
import math
# from selayer import SELayer

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=False, activation='relu'):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=False, activation='relu'):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):

        out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=False, activation='relu'):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=False, activation='relu'):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation)
        # self.conv = SELayer(num_filter*num_stages, num_filter, 1)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=False, activation='relu'):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='relu'):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation)
        # self.conv = SELayer(num_filter * num_stages, num_filter, 1)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0
