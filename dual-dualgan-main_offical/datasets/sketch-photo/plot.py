import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# 读取音频文件
def load_audio_file(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate


# 绘制语谱图并保存
def plot_and_save_spectrogram(signal, sample_rate, save_path):
    # 计算语谱图
    S = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 使用librosa绘制语谱图
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000)

    # 关闭坐标轴
    plt.axis('off')  # 或者使用 plt.gca().set_axis_off()

    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.show()  # 如果需要预览可以取消注释这一行


# 主程序
if __name__ == "__main__":
    # 替换这里的路径为你的音频文件路径
    file_path = './C_5.wav'
    # 指定保存图像的路径
    save_path = './C_5.jpg'

    signal, sample_rate = load_audio_file(file_path)
    plot_and_save_spectrogram(signal, sample_rate, save_path)