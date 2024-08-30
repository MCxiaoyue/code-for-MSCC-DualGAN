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
        # print(x.shape)
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(out_ch)
        self.dropout = dropout
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out


# 生成器 U-Net
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64):
        """
        定义生成器的网络结构
        :param in_ch: 输入数据的通道数
        :param out_ch: 输出数据的通道数
        :param ngf: 第一层卷积的通道数 number of generator's first conv filters
        """
        super(Generator, self).__init__()

        # U-Net encoder
        self.en1 = ConvBlock(in_ch, ngf, activation=False, batch_norm=False)
        self.en2 = ConvBlock(ngf, ngf * 2)
        self.en3 = ConvBlock(ngf * 2, ngf * 4)
        self.en4 = ConvBlock(ngf * 4, ngf * 8)
        self.en5 = ConvBlock(ngf * 8, ngf * 8)
        self.en6 = ConvBlock(ngf * 8, ngf * 8)
        self.en7 = ConvBlock(ngf * 8, ngf * 8)
        self.en8 = ConvBlock(ngf * 8, ngf * 8, batch_norm=False)

        # U-Net decoder
        self.de1 = DeconvBlock(ngf * 8, ngf * 8, dropout=True)
        self.de2 = DeconvBlock(ngf * 8 * 2, ngf * 8, dropout=True)
        self.de3 = DeconvBlock(ngf * 8 * 2, ngf * 8, dropout=True)
        self.de4 = DeconvBlock(ngf * 8 * 2, ngf * 8)
        self.de5 = DeconvBlock(ngf * 8 * 2, ngf * 4)
        self.de6 = DeconvBlock(ngf * 4 * 2, ngf * 2)
        self.de7 = DeconvBlock(ngf * 2 * 2, ngf)
        self.de8 = DeconvBlock(ngf * 2, out_ch, batch_norm=False)

    def forward(self, X):
        """
        生成器模块前向传播
        :param X: 输入生成器的数据
        :return: 生成器的输出
        """
        # Encoder
        en1_out = self.en1(X)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)
        en5_out = self.en5(en4_out)
        en6_out = self.en6(en5_out)
        en7_out = self.en7(en6_out)
        en8_out = self.en8(en7_out)

        # Decoder
        de1_out = self.de1(en8_out)
        de1_cat = torch.cat([de1_out, en7_out], dim=1)
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en6_out], 1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en5_out], 1)
        de4_out = self.de4(de3_cat)
        de4_cat = torch.cat([de4_out, en4_out], 1)
        de5_out = self.de5(de4_cat)
        de5_cat = torch.cat([de5_out, en3_out], 1)
        de6_out = self.de6(de5_cat)
        de6_cat = torch.cat([de6_out, en2_out], 1)
        de7_out = self.de7(de6_cat)
        de7_cat = torch.cat([de7_out, en1_out], 1)
        de8_out = self.de8(de7_cat)
        out = torch.nn.Tanh()(de8_out)

        return out
