import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设的SSIM值
ssim_values = [0.986165866375729, 0.9878221732224858, 0.9850715568413594, 0.9855056158187804, 0.986861876321338, 0.9858622532011335, 0.9834911839746178, 0.9877461716705939]



# 计算平均值
mean_ssim = np.mean(ssim_values)

# 创建箱型图
plt.figure(figsize=(8, 6))
sns.boxplot(y=ssim_values)
plt.axhline(mean_ssim, color='r', linestyle='--')  # 添加平均值的水平线
plt.text(mean_ssim + 0.01, mean_ssim, f'Mean: {mean_ssim:.2f}', verticalalignment='center', color='red')
plt.ylabel('SSIM Value')
plt.title('Boxplot of SSIM Values with Mean')

# 显示图形
plt.show()