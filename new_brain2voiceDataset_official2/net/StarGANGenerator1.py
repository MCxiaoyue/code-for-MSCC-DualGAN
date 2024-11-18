import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64, n_res_blocks=6, n_upsamplings=2):
        super(Generator, self).__init__()

        # Initial convolution layers
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks += [ResidualBlock(in_features)]
        model += res_blocks

        # Upsampling
        upsample_blocks = []
        for _ in range(n_upsamplings):
            upsample_blocks += [
                nn.ConvTranspose2d(in_features, in_features // 2, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=False),
                nn.InstanceNorm2d(in_features // 2),
                nn.ReLU(inplace=True)]
            in_features = in_features // 2
        model += upsample_blocks

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_features, out_channels, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        y = self.model(x)
        return y