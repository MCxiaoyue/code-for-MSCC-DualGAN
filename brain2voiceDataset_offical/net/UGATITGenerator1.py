import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualAttentionBlock, self).__init__()
        self.residual_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        out = self.residual_block(x)
        return x + out


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        energy = torch.bmm(query.view(*query.size()[:2], -1).permute(0, 2, 1), key.view(*key.size()[:2], -1))
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(value.view(*value.size()[:2], -1), attention.permute(0, 2, 1))
        out = out.view(*value.size())
        out = self.gamma * out + x

        return out


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=4, n_attention_blocks=1):
        super(Generator, self).__init__()

        # Initial convolution block
        self.down1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        # Downsampling
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualAttentionBlock(256) for _ in range(n_residual_blocks)])

        # Attention blocks
        self.attention_blocks = nn.Sequential(*[AttentionBlock(256) for _ in range(n_attention_blocks)])

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        # Output layer
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.down1(x)
        out = self.down2(out)
        out = self.down3(out)
        out = self.res_blocks(out)
        out = self.attention_blocks(out)
        out = self.up1(out)
        out = self.up2(out)
        out = self.out(out)
        return out
