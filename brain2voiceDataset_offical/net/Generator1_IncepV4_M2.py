import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        if self.activation:
            x = self.lrelu(x)
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(out_ch)
        self.dropout = dropout
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(x)
        x = self.deconv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf=64):
        super(Generator, self).__init__()

        # U-Net encoder
        self.en1 = ConvBlock(in_ch, ngf, activation=False, batch_norm=False)
        self.en2 = ConvBlock(ngf, ngf * 2)
        self.en3 = ConvBlock(ngf * 2, ngf * 4)

        # Bottleneck layer
        self.bottle_neck = ConvBlock(ngf * 4, ngf * 8, batch_norm=False)

        # U-Net decoder
        self.de1 = DeconvBlock(ngf * 8, ngf * 4, dropout=True)
        self.de2 = DeconvBlock(ngf * 8, ngf * 2)
        self.de3 = DeconvBlock(ngf * 4, ngf)

        # Final layer to get the output with desired channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, X):
        # Encoder
        en1_out = self.en1(X)
        en2_out = self.en2(en1_out)
        en3_out = self.en3(en2_out)

        # Bottleneck
        bottle_neck_out = self.bottle_neck(en3_out)

        # Decoder
        de1_out = self.de1(bottle_neck_out)
        de1_cat = torch.cat([de1_out, en3_out], dim=1)
        de2_out = self.de2(de1_cat)
        de2_cat = torch.cat([de2_out, en2_out], dim=1)
        de3_out = self.de3(de2_cat)
        de3_cat = torch.cat([de3_out, en1_out], dim=1)

        # Final output
        out = self.final(de3_cat)

        return out