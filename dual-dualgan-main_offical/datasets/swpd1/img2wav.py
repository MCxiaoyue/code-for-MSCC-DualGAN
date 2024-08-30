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


for i in range(91, 101):
    # # 使用函数
    image_to_wav("E:\\dual-dualgan-main_offical\\test\\swpd1-img_sz_256-fltr_dim_64-L1-lambda_BC_1000.0_1000.0\\B\\B_" + str(i) + "_B2C.jpg", "./BtoC_"+str(i)+".wav")
    # image_to_wav("./test/C/C_"+str(i)+".png", "./C_" + str(i) + ".wav")