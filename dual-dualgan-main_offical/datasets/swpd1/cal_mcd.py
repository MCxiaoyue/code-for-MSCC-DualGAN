from pymcd.mcd import Calculate_MCD
import numpy as np


# 初始化一个变量来存储MCD值总和
total_mcd = 0

# 创建一个计数器来记录成功计算MCD值的次数
count = 0

mcd_values = []

for i in range(91, 101):
    # 创建 Calculate_MCD 类的实例
    # "plain" 模式计算最直接的 MCD
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")

    # 两个输入文件路径，分别对应参考（真实）音频和合成音频
    try:
        mcd_value = mcd_toolbox.calculate_mcd("./C_" + str(i) + ".wav", "./BtoC_" + str(i) + ".wav")
        print(i)
        print("MCD Value:", mcd_value)
        print('==============================')

        # 累加MCD值并将其添加到列表中
        total_mcd += mcd_value
        mcd_values.append(mcd_value)
        count += 1  # 增加计数器
    except Exception as e:
        print(f"An error occurred: {e}")

# 计算平均MCD值
if count > 0:
    average_mcd = total_mcd / count
    print(f"Average MCD Value: {average_mcd}")

    # 使用numpy计算标准差
    std_dev = np.std(mcd_values)
    print(f"Standard Deviation of MCD Values: {std_dev}")
else:
    print("No MCD values were calculated.")