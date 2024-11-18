from PIL import Image
import numpy as np
from scipy.io import wavfile

def image_to_wav(image_path, output_wav_path):
    # 读取图像并调整尺寸
    img = Image.open(image_path).resize((256, 256)).convert('L')

    # 提取像素值
    img_data = np.array(img)

    # 将像素值缩放到音频样本的范围 (-32768 to 32767) 并转换为 int16 类型
    audio_samples = (img_data - 128) * 256  # 缩放以匹配 16-bit PCM 音频
    audio_samples = audio_samples.astype(np.int16)

    # 设置音频参数
    sample_rate = 44100  # 采样率

    # 保存为 .wav 文件
    wavfile.write(output_wav_path, sample_rate, audio_samples.flatten())


for i in [5, 10, 21, 26, 37, 42, 53, 58, 69, 74]:
    # # 使用函数
    image_to_wav("./test_AtoB_results/AtoB_"+str(i)+".PNG", "./AtoB_"+str(i)+".wav")
    image_to_wav("./test/B/B_"+str(i)+".png", "./B_" + str(i) + ".wav")