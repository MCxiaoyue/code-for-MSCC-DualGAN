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


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64):
        super(Generator, self).__init__()

        # U-Net encoder (reduced to 4 layers)
        self.en1 = ConvBlock(in_ch, ngf, activation=False, batch_norm=False)
        self.en2 = ConvBlock(ngf, ngf * 2)
        self.en3 = ConvBlock(ngf * 2, ngf * 4)
        self.en4 = ConvBlock(ngf * 4, ngf * 8, batch_norm=False)

        # U-Net decoder (reduced to 4 layers)
        self.de1 = DeconvBlock(ngf * 8, ngf * 4, dropout=True)
        self.de2 = DeconvBlock(ngf * 4 * 2, ngf * 2)
        self.de3 = DeconvBlock(ngf * 2 * 2, ngf)
        self.de4 = DeconvBlock(ngf * 2, out_ch, batch_norm=False)

    def forward(self, X):
        # Encoder
        en1_out = self.en1(X)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)
        en4_out = self.en4(en3_out)

        # Decoder
        de1_out = self.de1(en4_out)
        de1_cat = torch.cat([de1_out, en3_out], dim=1)
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en2_out], 1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en1_out], 1)
        de4_out = self.de4(de3_cat)
        out = torch.nn.Tanh()(de4_out)

        return out
