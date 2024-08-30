import cv2
import numpy as np


def crop_and_concat_images(image1_path, image2_path, output_path, ratio=3 / 5):
    # 读取两张输入图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images could not be read. Please check the paths.")

    height, width, _ = img1.shape

    # 计算裁剪高度
    crop_height1 = int(height * ratio)
    crop_height2 = height - crop_height1

    # 裁剪图像
    cropped_img1 = img1[:crop_height1, :]
    cropped_img2 = img2[-crop_height2:, :]

    # 创建一个与原图等宽、高为两图裁剪高度之和的空白图像
    concatenated_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将裁剪后的图像拼接到空白图像上
    concatenated_image[:crop_height1, :] = cropped_img1
    concatenated_image[crop_height1:, :] = cropped_img2

    # 保存拼接后的图像
    cv2.imwrite(output_path, concatenated_image)


# 使用循环处理所有图像
for i in range(91, 101):
    # 指定输入图像路径和输出路径
    image1_path = "E:\\dual-dualgan-main_offical\\datasets\\swpd1\\test\\A\\A_" + str(
        i) + ".png"
    image2_path = "E:\\dual-dualgan-main_offical\\datasets\\swpd1\\test\\C\\B_" + str(
        i) + ".png"
    output_image_path = "E:\\dual-dualgan-main_offical\\datasets\\swpd1\\test\\B\\B_" + str(
        i) + ".png"

    # 调用函数进行图像裁剪和拼接
    crop_and_concat_images(image1_path, image2_path, output_image_path)