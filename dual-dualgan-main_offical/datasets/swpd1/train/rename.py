import os
import shutil

# 指定包含图片的文件夹路径
folder_path = './C'

# 获取文件夹中所有 .png 文件
png_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and f.startswith('B_')]

# 重命名每个文件
for old_filename in png_files:
    # 提取数字部分
    number_part = old_filename.split('_')[1].split('.')[0]

    # 构建新的文件名
    new_filename = f'C_{number_part}.png'

    # 完整的旧文件路径和新文件路径
    old_filepath = os.path.join(folder_path, old_filename)
    new_filepath = os.path.join(folder_path, new_filename)

    # 重命名文件
    shutil.move(old_filepath, new_filepath)

print("所有文件已重命名完成。")