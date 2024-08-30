import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        )

    def forward(self, x):
        return self.main(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(DeconvBlock, self).__init__()
        self.use_dropout = use_dropout
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.BatchNorm2d(out_channels) if not use_dropout else nn.Identity()
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.encoder = nn.ModuleList([
            ConvBlock(in_channels, ngf, False),
            ConvBlock(ngf, ngf * 2),
            ConvBlock(ngf * 2, ngf * 4),
            ConvBlock(ngf * 4, ngf * 8),
            ConvBlock(ngf * 8, ngf * 8),
            ConvBlock(ngf * 8, ngf * 8),
            ConvBlock(ngf * 8, ngf * 8),
            ConvBlock(ngf * 8, ngf * 8, False)
        ])

        self.decoder = nn.ModuleList([
            DeconvBlock(ngf * 8 * 2, ngf * 8, True),  # 注意这里输入通道数加倍
            DeconvBlock(ngf * 8 * 2, ngf * 8, True),  # 输入通道数加倍
            DeconvBlock(ngf * 8 * 2, ngf * 8, True),  # 输入通道数加倍
            DeconvBlock(ngf * 8 * 2, ngf * 8),        # 输入通道数加倍
            DeconvBlock(ngf * 8 * 2, ngf * 4),        # 输入通道数加倍
            DeconvBlock(ngf * 4 * 2, ngf * 2),        # 输入通道数加倍
            DeconvBlock(ngf * 2 * 2, ngf),            # 输入通道数加倍
            DeconvBlock(ngf, out_channels, False) # 最后一层不需要输入通道数加倍
        ])

    def forward(self, x):
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)

        for i, layer in enumerate(self.decoder):
            if i < len(self.decoder) - 1:  # Exclude the last layer from skip connections
                x = torch.cat([x, encoder_outputs[-(i + 1)]], dim=1)
            x = layer(x)



        return torch.tanh(x)