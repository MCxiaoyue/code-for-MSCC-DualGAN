import cv2
import sys
import shutil
import subprocess
import scipy.signal as signal
import librosa
import soundfile as sf
import copy
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


sr = 11025  # Sample rate. 设置采样率为 11025 Hz。采样率表示每秒对信号的采样次数，这里是音频信号的采样率。
n_fft = 2206  # fft points (samples) 设置 FFT（快速傅里叶变换）的点数为 2048，用于频谱分析和变换。 time will prove that对应2206
frame_shift = 0.0125  # seconds 设置帧移（frame shift）为 0.0125 秒。在音频处理中，通常将音频信号分成短时间窗口进行处理，帧移表示相邻两个窗口的时间间隔。
frame_length = 0.15  # seconds : 设置帧长（frame length）为 0.05 秒。这是短时傅里叶变换（STFT）所用的每个窗口的长度。
hop_length = int(sr*frame_shift)   # samples. 计算帧移的样本数。采样率乘以帧移得到每个窗口之间的样本数。
win_length = int(sr*frame_length)  # samples.  计算帧长的样本数。采样率乘以帧长得到每个窗口的样本数。
n_mels = 448  # 设置生成 Mel 滤波器组的数量为 80。Mel 频率倒谱系数（MFCC）是一种常用的音频特征提取方法，这里设置了用于提取 MFCC 的 Mel 滤波器组数量。
power = 1.2  # Exponent for amplifying the predicted magnitude 设置用于放大预测幅度的指数。在信号还原时，预测的幅度可能需要进行一定程度的放大。
n_iter = 100  # Number of inversion iterations 设置反变换的迭代次数。在进行信号还原时，可能需要进行多次迭代以获得更准确的结果。
preemphasis = .97  # or None设置预加重滤波器的系数。预加重滤波器用于突出高频部分，帮助改善信噪比。
max_db = 100  # 设置能量值的上限。在进行能量值计算时，可能会对能量进行限制，确保其不超过该阈值。
ref_db = 20  # 设置参考能量值的阈值。用于计算相对能量的参考值。
top_db = 15  # 设置能量的上限范围。用于对信号的能量范围进行限制或压缩，确保信号的动态范围在可接受的范围内。


def melspectrogram2wav(mel):
    '''# Generate wave file from spectrogram 将梅尔频谱图转换回音频波形。'''
    # transpose 转置输入的梅尔频谱图 (mel)，将其形状变为 (n_mels, T)。
    mel = mel.T  # (n_mels, T)

    # de-noramlize 对梅尔频谱图进行反归一化操作，将其值限制在 [0, 1] 范围内，然后再进行反归一化变换，将其还原为原始的对数刻度。
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)  # 将反归一化后的对数梅尔频谱图转换为幅度谱图。
    m = _mel_to_linear_matrix(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mag = np.dot(m, mel)  # 使用 _mel_to_linear_matrix 函数将梅尔频谱图转换为线性频谱图，然后通过矩阵乘法得到对应的幅度谱图 (mag)。

    # wav reconstruction 使用 Griffin-Lim 算法将幅度谱图还原为波形。
    wav = griffin_lim(mag)

    # de-preemphasis 进行反预加重操作，通过 IIR 滤波器将预加重效果还原。
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim 对还原的波形进行修剪，去除静音部分。
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)  # 返回最终的音频波形，并将数据类型转换为 np.float32


def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # 使用 Librosa 库的 mel 函数生成梅尔滤波器矩阵，该矩阵用于将线性频谱图转换为梅尔频谱图。
    m_t = np.transpose(m)  # 对梅尔滤波器矩阵进行转置，得到其转置矩阵
    p = np.matmul(m, m_t)  # 计算梅尔滤波器矩阵与其转置矩阵的乘积。
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]  # 计算对角线矩阵的对角元素，每个元素为对应列和的倒数。如果倒数的绝对值小于1.0e-8则直接使用原始值。
    return np.matmul(m_t, np.diag(d))  # 将转置梅尔滤波器矩阵与由对角元素构成的对角矩阵相乘，得到最终的转换矩阵。


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)  # 对输入的谱图进行深拷贝，将其保存为 X_best。
    for i in range(n_iter):  # 对于指定的迭代次数 n_iter，执行以下步骤。
        X_t = invert_spectrogram(X_best)  # 调用 invert_spectrogram 函数将当前的谱图 X_best 转换为时域信号。
        est = librosa.stft(X_t, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # 使用 Librosa 库的 stft 函数计算当前时域信号的短时傅里叶变换。
        phase = est / np.maximum(1e-8, np.abs(est))  # 计算相位信息，避免除零错误。
        X_best = spectrogram * phase  # 更新 X_best，将其乘以相位信息。
    X_t = invert_spectrogram(X_best)  # 将更新后的 X_best 转换回时域信号。
    y = np.real(X_t)  # 提取时域信号的实部，得到最终的音频波形。

    return y  # 返回 Griffin-Lim 算法还原的音频波形。


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    使用 Librosa 库的 istft 函数将输入的谱图转换为时域信号，采用汉明窗口。
    '''
    return librosa.istft(spectrogram, hop_length=hop_length, win_length=win_length, window="hann")


for name in range(1, 63, 10):
    # name = "B_"+str(name)
    # input_image = Image.open("E:\\new_brain2voiceDataset_official2\\timit-1\\test\\B\\"+str(name)+".png")

    name = "AtoB_" + str(name)
    input_image = Image.open("E:\\new_brain2voiceDataset_official2\\timit-1\\test_AtoB_results\\" + str(name) + ".PNG")


    # 转换为8位深度图像
    output_image = input_image.convert('L')
    output_image.save("./Recon_"+str(name)+".png")

    # 加载图像文件并转换为数组
    mel_img = mpimg.imread("./Recon_"+str(name)+".png")
    mel_data = mel_img.astype(float)

    word = 'res_'
    number1 = 120
    number2 = 448
    mel_data = mel_data.reshape(-1)
    mel_data = mel_data[
               :int(number1) * int(number2)]  # n_mel=82时  that she(4018)    n_mel=80时 (3920)   n_mel=300时 (14700)
    mel_data = mel_data.reshape(int(number1),
                                int(number2))  # n_mel=82时 that she（49， 82）  n_mel=80时 (49,80) n_mel=300时 (49,300)

    # 图像通常会添加颜色映射，所以可能需要进一步处理才能得到原始的梅尔频谱图数据
    # 具体处理方式取决于原始保存时的颜色映射和数据格式
    # 假设它是灰度图像且存储了归一化后的梅尔频谱图数据
    mel_spectrogram = mel_data.squeeze()  # 如果图像是一维的，取消多余的维度
    # 或者如果是彩色图像并且只有一层颜色代表梅尔频谱，则可能需要 mel_spectrogram = mel_data[:,:,0]

    # 将图像数据转换为正确维度的梅尔频谱数组
    # 注意：这里的维度转换假设图像的宽度代表梅尔频谱的时间帧数，高度代表梅尔频段数
    # 可能需要翻转数组以适应之前的代码结构
    # mel_spectrogram = mel_spectrogram.T

    # 现在有了梅尔频谱数组，可以调用之前定义好的逆转换函数
    wav1 = melspectrogram2wav(mel_spectrogram)

    # 最后，将音频波形写入到.wav文件
    sf.write("./" +str(name)+ ".wav", wav1, sr)



