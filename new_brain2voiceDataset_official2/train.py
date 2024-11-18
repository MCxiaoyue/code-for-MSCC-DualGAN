import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision.utils import save_image
from net import Generator1
from net import Discriminator1
from util import parseArgs
from util import loader
from util.logger import my_log
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 训练判别器A ##
def D_A_train(D_A: Discriminator1, G_B: Generator1, X, Y, BCELoss, optimizer_D):
    # print(X.shape)
    # print(Y.shape)
    # print('==========================================')
    """
    训练判别器A
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器的损失值
    """
    x = X.to(device)
    y = Y.to(device)
    # 梯度初始化为0
    yx = torch.cat([y, x], dim=1)
    D_A.zero_grad()
    # 在真数据上
    D_output_r = D_A(yx).squeeze()
    # 在假数据上
    G_B_output = G_B(y)
    yx_fake = torch.cat([y, G_B_output], dim=1)
    D_output_f = D_A(yx_fake).squeeze()
    if random.random() < 0.1:
        D_real_loss = BCELoss(D_output_r, torch.zeros(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    else:
        D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))
    # 反向传播并优化
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()


## 训练判别器B ##
def D_B_train(D_B: Discriminator1, G_A: Generator1, X, Y, BCELoss, optimizer_D):
    """
    训练判别器B
    :param BCELoss: 二分交叉熵损失函数
    :param optimizer_D: 判别器优化器
    :return: 判别器的损失值
    """
    # 标签转实物（右转左）
    x = X.to(device)
    y = Y.to(device)
    # 梯度初始化为0
    D_B.zero_grad()
    # 在真数据上
    xy = torch.cat([x, y], dim=1)
    D_output_r = D_B(xy).squeeze()
    # 在假数据上
    G_A_output = G_A(x)
    xy_fake = torch.cat([x, G_A_output], dim=1)
    D_output_f = D_B(xy_fake).squeeze()
    if random.random() < 0.1:
        D_real_loss = BCELoss(D_output_r, torch.zeros(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
    else:
        D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
        D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))

    # 反向传播并优化
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()

    return D_loss.data.item()


## 训练生成器 ##
def G_train(D_A: Discriminator1, D_B: Discriminator1, G_A: Generator1, G_B: Generator1, X, Y, BCELoss, L1, optimizer_G,
            lamb=100):
    """
    训练生成器
    :param BCELoss: 二分交叉熵损失函数
    :param L1: L1正则化函数
    :param optimizer_G: 生成器优化器
    :param lamb: L1正则化的权重
    :return: 生成器的损失值
    """

    x = X.to(device)
    y = Y.to(device)

    # 梯度初始化为0
    G_A.zero_grad()
    G_B.zero_grad()
    # 在假数据上
    G_A_output = G_A(x)
    xy_fake = torch.cat([x, G_A_output], dim=1)
    D_B_output_f = D_B(xy_fake).squeeze()
    G_A_BCE_loss = BCELoss(D_B_output_f, torch.ones(D_B_output_f.size()).to(device))
    G_A_L1_Loss = L1(G_A_output, y)
    # 反向传播并优化
    G_A_loss = G_A_BCE_loss + lamb * G_A_L1_Loss
    # 在假数据上
    G_B_output = G_B(y)
    yx_fake = torch.cat([y, G_B_output], dim=1)
    D_A_output_f = D_A(yx_fake).squeeze()
    G_B_BCE_loss = BCELoss(D_A_output_f, torch.ones(D_A_output_f.size()).to(device))
    G_B_L1_Loss = L1(G_B_output, x)
    # 反向传播并优化
    G_B_loss = G_B_BCE_loss + lamb * G_B_L1_Loss
    G_loss = G_A_loss + G_B_loss
    G_loss.backward(retain_graph=True)
    optimizer_G.step()

    return G_loss.data.item()


#主函数：训练DualGan
def main():
    # 加载训练数据
    args = parseArgs.parseArgs()
    logger = my_log()
    save_path = args.save_path
    data_path = args.data_path
    trainB_path = args.trainB
    trainA_path = args.trainA
    batch_size = args.batch_size
    image_size = args.image_size
    # GT_loader = loader.loadData(data_path, GT_path, image_size=image_size, batch_size=batch_size)
    # hazy_loader = loader.loadData(data_path, hazy_path, image_size=image_size, batch_size=batch_size)

    trainB_loader = loader.loadData(data_path, trainB_path,  batch_size=batch_size, img_size=image_size)
    trainA_loader = loader.loadData(data_path, trainA_path, batch_size=batch_size, img_size=image_size)

    # 定义结构参数
    in_ch, out_ch = 3, 3  # 输入输出图片通道数
    ngf, ndf = 64, 64  # 生成数、判别器第一层卷积通道数

    # 定义训练参数
    lr_G, lr_D = 0.00005, 0.00001  # G、D的学习速率
    lamb = 100  # 在生成器的目标函数中L1正则化的权重
    epochs = 250  # 训练迭代次数

    # 声明生成器、判别器
    G_A = Generator1.Generator(in_ch, out_ch, ngf).to(device)  # 生成B
    G_B = Generator1.Generator(in_ch, out_ch, ngf).to(device)  # 生成A
    D_A = Discriminator1.Discriminator(in_ch + out_ch, ndf).to(device)  # 鉴别A
    D_B = Discriminator1.Discriminator(in_ch + out_ch, ndf).to(device)  # 鉴别B

    # 目标函数 & 优化器
    BCELoss = nn.BCELoss().to(device)
    L1 = nn.L1Loss().to(device)
    optimizer_G = optim.RMSprop(itertools.chain(G_A.parameters(), G_B.parameters()), lr=lr_G, alpha=0.99, eps=1e-08)
    optimizer_D_A = optim.RMSprop(D_A.parameters(), lr=lr_D, alpha=0.99, eps=1e-08)
    optimizer_D_B = optim.RMSprop(D_B.parameters(), lr=lr_D, alpha=0.99, eps=1e-08)

    # 输入数据
    X, labelx = next(iter(trainA_loader))
    Y, labely = next(iter(trainB_loader))

    g_a = G_A(X.to(device))  # A->B
    g_b = G_B(Y.to(device))  # B->A
    logger.info(g_a.size())
    save_image(X, save_path + labelx[0].split('.')[0] + '_input.png')
    save_image(Y, save_path + labely[0].split('.')[0] + '_ground-truth.png')
    save_image(g_a, save_path + labelx[0].split('.')[0] + '_sample_trainB_0.png')
    save_image(g_b, save_path + labely[0].split('.')[0] + '_sample_trainA_0.png')

    # 开始训练
    G_A.train()
    G_B.train()
    D_A.train()
    D_B.train()
    D_A_Loss, D_B_Loss, G_Loss, Epochs = [], [], [], range(1, epochs + 1)
    for epoch in range(epochs):
        D_A_losses, D_B_losses, G_losses, batch, d_a_l, d_b_l, g_l = [], [], [], 0, 0, 0, 0
        trainB_iter = iter(trainB_loader)
        trainA_iter = iter(trainA_loader)
        for i in range(0, len(trainB_loader)):
            X, _ = next(trainA_iter)
            Y, _ = next(trainB_iter)
            batch += 1
            # 训练Discriminator并保存loss
            D_A_losses.append(D_A_train(D_A, G_B, X, Y, BCELoss, optimizer_D_A))
            D_B_losses.append(D_B_train(D_B, G_A, X, Y, BCELoss, optimizer_D_B))
            # 训练Generator
            G_losses.append(G_train(D_A, D_B, G_A, G_B, X, Y, BCELoss, L1, optimizer_G, lamb))
            if batch % 10 == 1:
                # 打印每十次batch的平均loss
                d_a_l, d_b_l, g_l = np.array(D_A_losses).mean(), np.array(D_B_losses).mean(), np.array(G_losses).mean()
                print('[%d / %d]: batch#%d loss_d_a= %.3f  loss_d_b= %.3f  loss_g= %.3f' %
                      (epoch + 1, epochs, batch, d_a_l, d_b_l, g_l))
        # 测试每十次epoch的生成效果
        if (epoch + 1) % 10 == 0:
            X, labelx = next(iter(trainA_loader))
            Y, labely = next(iter(trainB_loader))
            g_a = G_A(X.to(device))
            g_b = G_B(Y.to(device))
            save_image(g_a, save_path + labelx[0].split('.')[0] + '_sample_AtoB_' + str(epoch + 1) + '.jpg')
            save_image(g_b, save_path + labely[0].split('.')[0] + '_sample_BtoA_' + str(epoch + 1) + '.jpg')
        # 保存每次epoch的loss
        D_A_Loss.append(d_a_l)
        D_B_Loss.append(d_b_l)
        G_Loss.append(g_l)
    print("Done!")

    # 保存训练结果
    torch.save(G_A, './'+data_path+'/checkpoint/generator_a.pkl')
    torch.save(G_B, './'+data_path+'/checkpoint/generator_b.pkl')
    torch.save(D_A, './'+data_path+'/checkpoint/discriminator_a.pkl')
    torch.save(D_B, './'+data_path+'/checkpoint/discriminator_b.pkl')
    # 画出loss图
    # G的loss因为包含L1 相比D的loss太大了 画图效果不好 所以除以100
    plt.plot(Epochs, D_A_Loss, label='Discriminator_A Losses')
    plt.plot(Epochs, D_B_Loss, label='Discriminator_B Losses')
    plt.plot(Epochs, np.array(G_Loss), label='Generator Losses')
    plt.legend()
    plt.savefig(save_path + 'loss.png')
    plt.show()


# 运行
if __name__ == '__main__':
    main()
