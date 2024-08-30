import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 假设的SSIM值
ssim_values = [0.986165866375729, 0.9878221732224858, 0.9850715568413594, 0.9855056158187804,
               0.986861876321338, 0.9858622532011335, 0.9834911839746178, 0.9877461716705939]

ssim_values1 = [0.2947599644164905, 0.2502909449246726, 0.9785014860811, 0.3455560781686128,
                0.2904759120652104, 0.9820109446326637, 0.29289187941552053, 0.9805896784293409]

# 将两组SSIM值合并
all_ssim_values = np.concatenate((ssim_values, ssim_values1))

# 创建标签列表（用于区分哪一组数据）
labels = ['Group 1'] * len(ssim_values) + ['Group 2'] * len(ssim_values1)

# 创建DataFrame
data = pd.DataFrame({'SSIM': all_ssim_values, 'Group': labels})

# 使用Seaborn创建箱型图
plt.figure(figsize=(10, 8))  # 调整图表大小
sns.boxplot(x='Group', y='SSIM', data=data)
plt.axhline(np.mean(ssim_values), color='r', linestyle='--', label=f'平均值: Group 1 - {np.mean(ssim_values):.2f}')
plt.axhline(np.mean(ssim_values1), color='b', linestyle='--', label=f'平均值: Group 2 - {np.mean(ssim_values1):.2f}')
plt.legend()
plt.ylim(0.3, 1)  # 设置Y轴范围
plt.ylabel('SSIM 值')
plt.title('SSIM 值的箱型图及平均值')
plt.show()