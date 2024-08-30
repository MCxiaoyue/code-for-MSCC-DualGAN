import torch
from torch import nn

class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):

        out = self.conv(x)

        if self.batch_norm:
            out = self.bn(out)

        if self.activation:
            return self.lrelu(out)
        else:
            return out




# 辨别器 PatchGAN
class Discriminator(nn.Module):
    def __init__(self, in_ch, ndf=64):
        """
        定义判别器的网络结构
        :param in_ch: 输入数据的通道数
        :param ndf: 第一层卷积的通道数 number of discriminator's first conv filters
        """
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(in_ch, ndf * 4, batch_norm=False)
        # self.conv2 = ConvBlock(ndf, ndf * 2)
        # self.conv3 = ConvBlock(ndf * 2, ndf * 4)
        self.conv4 = ConvBlock(ndf * 4, ndf * 2)
        self.conv5 = ConvBlock(ndf * 2, ndf, activation=False, batch_norm=False)


    def forward(self, x):
        """
        判别器模块正向传播
        :param x: 输入判别器的数据
        :return: 判别器的输出
        """
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)

        return out
