import torch
import torch.nn.functional as F
import torch.nn as nn

# 2. 设计基础模型
class InceptionA(torch.nn.Module):
    def __init__(self, inChannels):  # inChannels表输入通道数
        super(InceptionA, self).__init__()
        # 2.1 第一层池化 + 1*1卷积
        self.branch1_1x1 = nn.Conv2d(in_channels=inChannels,  # 输入通道
                                     out_channels=int(inChannels/4),  # 输出通道
                                     kernel_size=1)  # 卷积核大小1*1
        # 2.2 第二层1*1卷积
        self.branch2_1x1 = nn.Conv2d(inChannels, int(inChannels/4), kernel_size=1)

        # 2.3 第三层
        self.branch3_1_1x1 = nn.Conv2d(inChannels, int(inChannels/8), kernel_size=1)
        self.branch3_2_5x5 = nn.Conv2d(int(inChannels/8), int(inChannels/4), kernel_size=3, padding=1)
        # padding=2,因为要保持输出的宽高保持一致

        # 2.4 第四层
        self.branch4_1_1x1 = nn.Conv2d(inChannels, int(inChannels/16), kernel_size=1)
        self.branch4_2_3x3 = nn.Conv2d(int(inChannels/16), int(inChannels/8), kernel_size=3, padding=1)
        self.branch4_3_3x3 = nn.Conv2d(int(inChannels/8), int(inChannels/4), kernel_size=3, padding=1)

    def forward(self, X_input):
        # 第一层
        branch1_pool = F.avg_pool2d(X_input,  # 输入
                                    kernel_size=3,  # 池化层的核大小3*3
                                    stride=1,  # 每次移动一步
                                    padding=1)
        branch1 = self.branch1_1x1(branch1_pool)
        # 第二层
        branch2 = self.branch2_1x1(X_input)
        # 第三层
        branch3_1 = self.branch3_1_1x1(X_input)
        branch3 = self.branch3_2_5x5(branch3_1)
        # 第四层
        branch4_1 = self.branch4_1_1x1(X_input)
        branch4_2 = self.branch4_2_3x3(branch4_1)
        branch4 = self.branch4_3_3x3(branch4_2)
        # 输出
        output = [branch2, branch3, branch4, branch1]
        # (batch_size, channel, w, h)   dim=1: 即安装通道进行拼接。
        # eg: (1, 2, 3, 4) 和 （1, 4, 3, 4）按照dim=1拼接，则拼接后的shape为（1, 2+4, 3,  4）
        return torch.cat(output, dim=1)
# InceptionB模块
class InceptionB(torch.nn.Module):
    def __init__(self, inChannels):
        super(InceptionB, self).__init__()

        # 3.1 第一层1*1卷积
        self.branch1_1x1 = nn.Conv2d(in_channels=inChannels,
                                     out_channels=int(inChannels / 4),
                                     kernel_size=1)

        # 3.2 第二层1*7卷积 + 7*1卷积
        self.branch2_1x1 = nn.Conv2d(in_channels=inChannels,
                                     out_channels=int(inChannels / 4),
                                     kernel_size=1)

        # 3.3 第三层先进行1*1卷积（增加通道数），然后是1*7卷积（padding=0）+ 7*1卷积（padding=0）
        self.branch3_1x1_reduce = nn.Conv2d(inChannels, int(inChannels / 16), kernel_size=1)
        self.branch3_1x7 = nn.Conv2d(int(inChannels / 16), int(inChannels / 8), kernel_size=(1, 7), padding=(0, 3))
        self.branch3_7x1 = nn.Conv2d(int(inChannels / 8), int(inChannels / 4), kernel_size=(7, 1), padding=(3, 0))

        # 3.4 第四层1*1卷积后接3*3卷积，然后进行3*3最大池化
        self.branch4_1x1_reduce = nn.Conv2d(inChannels, int(inChannels / 64), kernel_size=1)
        self.branch4_1x7 = nn.Conv2d(int(inChannels / 64), int(inChannels / 32), kernel_size=(1, 7), padding=(0, 3))
        self.branch4_7x1 = nn.Conv2d(int(inChannels / 32), int(inChannels / 16), kernel_size=(7, 1), padding=(3, 0))
        self.branch4_1x7_2 = nn.Conv2d(int(inChannels / 16), int(inChannels / 8), kernel_size=(1, 7), padding=(0, 3))
        self.branch4_7x1_2 = nn.Conv2d(int(inChannels / 8), int(inChannels / 4), kernel_size=(7, 1), padding=(3, 0))

    def forward(self, X_input):
        # 第一层
        branch1 = self.branch1_1x1(X_input)

        # 第二层
        branch2_pool = F.avg_pool2d(X_input,  # 输入
                                    kernel_size=3,  # 池化层的核大小3*3
                                    stride=1,  # 每次移动一步
                                    padding=1)
        branch2 = self.branch2_1x1(branch2_pool)

        # 第三层
        branch3_intermediate = self.branch3_1x1_reduce(X_input)
        branch3_1x7 = self.branch3_1x7(branch3_intermediate)
        branch3 = self.branch3_7x1(branch3_1x7)

        # 第四层
        branch4_1x1 = self.branch4_1x1_reduce(X_input)
        branch4_1x7 = self.branch4_1x7(branch4_1x1)
        branch4_7x1 = self.branch4_7x1(branch4_1x7)
        branch4_1x7_2 = self.branch4_1x7_2(branch4_7x1)  # 将池化结果通过1x1卷积层调整通道数
        branch4 = self.branch4_7x1_2(branch4_1x7_2)  # 将3x3卷积的结果与调整通道数后的池化结果拼接

        # 输出
        output = [branch1, branch2, branch3, branch4]
        return torch.cat(output, dim=1)

# InceptionC模块
class InceptionC(torch.nn.Module):
    def __init__(self, inChannels):
        super(InceptionC, self).__init__()

        # 3.1 第一层 1x1卷积
        self.branch1_1x1 = nn.Conv2d(in_channels=inChannels, out_channels=int(inChannels / 4), kernel_size=1)

        # 3.2 第二层 1x1卷积后接3x3卷积
        self.branch2_1x1 = nn.Conv2d(in_channels=inChannels, out_channels=int(inChannels / 4), kernel_size=1)

        # 3.3 第三层 1x1卷积后接3x3卷积，再接3x3卷积（两次）
        self.branch3_1_1x1 = nn.Conv2d(in_channels=inChannels, out_channels=int(inChannels / 16), kernel_size=1)
        self.branch3_2_1x3 = nn.Conv2d(int(inChannels / 16), int(inChannels / 8), kernel_size=(1, 3), padding=(0, 1))
        self.branch3_3_3x1 = nn.Conv2d(int(inChannels / 16), int(inChannels / 8), kernel_size=(3, 1), padding=(1, 0))

        # 3.4 第四层 先进行3x3最大池化，后接1x1卷积
        self.branch4_1x1 = nn.Conv2d(in_channels=inChannels, out_channels=int(inChannels / 64), kernel_size=1)
        self.branch4_1x3 = nn.Conv2d(in_channels=int(inChannels / 64), out_channels=int(inChannels / 32), kernel_size=(1, 3), padding=(0, 1))
        self.branch4_3x1 = nn.Conv2d(in_channels=int(inChannels / 32), out_channels=int(inChannels / 16), kernel_size=(3, 1), padding=(1, 0))
        self.branch4_2_1x3 = nn.Conv2d(int(inChannels / 16), int(inChannels / 8), kernel_size=(1, 3), padding=(0, 1))
        self.branch4_2_3x1 = nn.Conv2d(int(inChannels / 16), int(inChannels / 8), kernel_size=(3, 1), padding=(1, 0))


    def forward(self, X_input):
        # 第一层
        branch1_pool = F.avg_pool2d(X_input,  # 输入
                                    kernel_size=3,  # 池化层的核大小3*3
                                    stride=1,  # 每次移动一步
                                    padding=1)
        branch1 = self.branch1_1x1(branch1_pool)

        # 第二层
        branch2 = self.branch2_1x1(X_input)

        # 第三层
        branch3_1_1x1 = self.branch3_1_1x1(X_input)
        branch3_2 = self.branch3_2_1x3(branch3_1_1x1)
        branch3_3 = self.branch3_3_3x1(branch3_1_1x1)
        branch3 = torch.cat((branch3_2, branch3_3), dim=1)

        # 第四层
        branch4_1x1 = self.branch4_1x1(X_input)
        branch4_1x3 = self.branch4_1x3(branch4_1x1)
        branch4_3x1 = self.branch4_3x1(branch4_1x3)

        branch4_2_1x3 = self.branch4_2_1x3(branch4_3x1)
        branch4_2_3x1 = self.branch4_2_3x1(branch4_3x1)
        branch4 = torch.cat((branch4_2_1x3, branch4_2_3x1), dim=1)

        # 输出
        output = [branch1, branch2, branch3, branch4]
        return torch.cat(output, dim=1)

# ReductionA模块
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super(ReductionA, self).__init__()

        # 分支1：先通过1x1卷积改变通道数，然后进行3x3卷积（步长为2）
        self.branch1_1x1 = nn.Conv2d(in_channels, int(in_channels / 8), kernel_size=1)
        self.branch1_3x3s1 = nn.Conv2d(int(in_channels / 8), int(in_channels / 4), kernel_size=3, padding=1)
        self.branch1_3x3s2 = nn.Conv2d(int(in_channels / 4), int(in_channels / 2), kernel_size=3, padding=1)

        # 分支2：与分支1结构相同，直接通过连续的1x1和3x3卷积层减少空间维度
        self.branch2_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch2_3x3s2 = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=3, padding=1)


    def forward(self, X_input):
        # 分支1
        branch1 = self.branch1_1x1(X_input)
        # print(f"X_input size:{X_input.shape}")
        # print(f"branch1 size:{branch1.shape}")
        branch1_0 = self.branch1_3x3s1(branch1)
        # print(f"branch1_0 size:{branch1_0.shape}")
        branch1 = self.branch1_3x3s2(branch1_0)
        # print(f"branch1 size:{branch1.shape}")

        # 分支2
        # branch2 = self.branch2_1x1(X_input)
        branch2 = self.branch2_maxpool(X_input)
        # print(f"X_input size:{X_input.shape}")
        # print(f"branch2_0 size:{branch2.shape}")
        branch2 = self.branch2_3x3s2(branch2)
        # print(f"branch2_1 size:{branch2.shape}")
        # print("==============================")

        # 合并两个分支的结果
        return torch.cat([branch2, branch1], dim=1)

# ReductionB模块
class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()

        # 分支1：先通过1x1卷积改变通道数，然后进行3x3卷积（步长为2）
        self.branch1_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch1_3x3s2 = nn.Conv2d(int(in_channels), int(in_channels / 2), kernel_size=3, padding=1)

        # 分支2：与分支1结构相同，直接通过连续的1x1和3x3卷积层减少空间维度
        self.branch2_1x1 = nn.Conv2d(in_channels, int(in_channels / 16), kernel_size=1)
        self.branch2_1x7 = nn.Conv2d(int(in_channels / 16), int(in_channels / 8), kernel_size=(1, 7), padding=(0, 3))
        self.branch2_7x1 = nn.Conv2d(int(in_channels / 8), int(in_channels / 4), kernel_size=(7, 1), padding=(3, 0))
        self.branch2_3x3 = nn.Conv2d(int(in_channels / 4), int(in_channels / 2), kernel_size=3, padding=1)

    def forward(self, X_input):
        # 分支1
        branch1 = self.branch1_pool(X_input)
        # print(f"X_input size:{X_input.shape}")
        # print(f"branch1_0 size:{branch1.shape}")
        branch1 = self.branch1_3x3s2(branch1)
        # print(f"branch1_1 size:{branch1.shape}")
        # print("================================")

        # 分支2
        branch2_1x1 = self.branch2_1x1(X_input)
        # print(f"X_input size:{X_input.shape}")
        # print(f"branch2_0 size:{branch2_1x1.shape}")
        branch2_1x7 = self.branch2_1x7(branch2_1x1)
        # print(f"branch2_1 size:{branch2_1x7.shape}")
        branch2_7x1 = self.branch2_7x1(branch2_1x7)
        # print(f"branch2_2 size:{branch2_7x1.shape}")
        branch2 = self.branch2_3x3(branch2_7x1)
        # print(f"branch2_3 size:{branch2.shape}")
        # print("================================")

        # 合并两个分支的结果
        return torch.cat([branch1, branch2], dim=1)

# 完整的InceptionV4模型
class InceptionV4(nn.Module):
    def __init__(self, inChannels):
        super(InceptionV4, self).__init__()

        # 前几层卷积网络（实际构建时请填充这部分）
        self.stem_network = nn.Conv2d(in_channels=inChannels,  # 输入通道
                                     out_channels=inChannels,  # 输出通道
                                     kernel_size=1)  # 卷积核大小1*1

        # 各种Inception模块堆叠
        self.inception_A = InceptionA(inChannels)
        self.inception_B = InceptionB(inChannels)
        self.inception_C = InceptionC(inChannels)
        self.reduction_A = ReductionA(inChannels)
        self.reduction_B = ReductionB(inChannels)

        # 最终输出层（没有激活函数）
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        # print(f"x size:{x.shape}")
        x = self.stem_network(x)
        # print(f"x_0 size:{x.shape}")
        x = self.inception_A(x)
        # print(f"x_1 size:{x.shape}")
        x = self.inception_B(x)
        # print(f"x_2 size:{x.shape}")
        x = self.inception_C(x)
        # print(f"x_3 size:{x.shape}")
        x = self.reduction_A(x)
        # print(f"x_4 size:{x.shape}")
        x = self.inception_B(x)  # 可以根据实际结构重复使用Inception模块
        # print(f"x_5 size:{x.shape}")
        x = self.inception_C(x)
        # print(f"x_6 size:{x.shape}")
        x = self.reduction_B(x)
        # print(f"x_7 size:{x.shape}")
        x = self.inception_C(x)
        # print(f"x_8 size:{x.shape}")
        x = self.avg_pool(x)
        # print(f"x_9 size:{x.shape}")
        # x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # print(f"x_10 size:{x.shape}")
        # x = self.fc(x)
        return x


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

        self.inceptionA_2 = InceptionV4(inChannels=(ngf * 8))
        # self.inceptionA_3 = InceptionV4(inChannels=(ngf * 8))
        # self.inceptionA_4 = InceptionV4(inChannels=(ngf * 8))
        # self.inceptionA_5 = InceptionV4(inChannels=(ngf * 8))
        # self.inceptionA_6 = InceptionV4(inChannels=(ngf * 4))
        # self.inceptionA_7 = InceptionV4(inChannels=(ngf * 2))
        # self.inceptionA_8 = InceptionV4(inChannels=(ngf))

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
        # # Encoder
        # en1_out = self.en1(X)
        # en2_out = self.en2(en1_out)
        # en3_out = self.en3(en2_out)
        # en4_out = self.en4(en3_out)
        # en5_out = self.en5(en4_out)
        # en6_out = self.en6(en5_out)
        # en7_out = self.en7(en6_out)
        # en8_out = self.en8(en7_out)
        #
        # # Decoder
        # de1_out = self.de1(en8_out)
        # de1_cat = torch.cat([de1_out, en7_out], dim=1)
        # de2_out = self.de2(de1_cat)
        # de2_cat = torch.cat([de2_out, en6_out], 1)
        # de3_out = self.de3(de2_cat)
        # de3_cat = torch.cat([de3_out, en5_out], 1)
        # de4_out = self.de4(de3_cat)
        # de4_cat = torch.cat([de4_out, en4_out], 1)
        # de5_out = self.de5(de4_cat)
        # de5_cat = torch.cat([de5_out, en3_out], 1)
        # de6_out = self.de6(de5_cat)
        # de6_cat = torch.cat([de6_out, en2_out], 1)
        # de7_out = self.de7(de6_cat)
        # de7_cat = torch.cat([de7_out, en1_out], 1)
        # de8_out = self.de8(de7_cat)
        # out = torch.nn.Tanh()(de8_out)

        # Encoder
        enc1 = self.en1(X)
        enc2 = self.en2(enc1)
        enc3 = self.en3(enc2)
        enc4 = self.en4(enc3)
        enc5 = self.en5(enc4)
        enc6 = self.en6(enc5)
        enc7 = self.en7(enc6)
        enc8 = self.en8(enc7)

        # Decoder with skip-connections
        dec1 = self.de1(enc8)
        dec1 = self.inceptionA_2(dec1)
        dec1 = torch.cat([dec1, enc7], 1)

        dec2 = self.de2(dec1)
        # dec2 = self.inceptionA_3(dec2)
        dec2 = torch.cat([dec2, enc6], 1)

        dec3 = self.de3(dec2)
        # dec3 = self.inceptionA_4(dec3)
        dec3 = torch.cat([dec3, enc5], 1)

        dec4 = self.de4(dec3)
        # dec4 = self.inceptionA_5(dec4)
        dec4 = torch.cat([dec4, enc4], 1)

        dec5 = self.de5(dec4)
        # dec5 = self.inceptionA_6(dec5)
        dec5 = torch.cat([dec5, enc3], 1)

        dec6 = self.de6(dec5)
        # dec6 = self.inceptionA_7(dec6)
        dec6 = torch.cat([dec6, enc2], 1)

        dec7 = self.de7(dec6)
        # dec7 = self.inceptionA_8(dec7)
        dec7 = torch.cat([dec7, enc1], 1)

        dec8 = self.de8(dec7)
        out = torch.nn.Tanh()(dec8)

        return out
