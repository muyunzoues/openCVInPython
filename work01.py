import numpy as np
import cv2

def median_filter_color(image, kernel_size):
    """
    中值滤波函数（彩色图像）
    :param image: 输入图像
    :param kernel_size: 滤波器大小
    :return: 滤波后的图像
    """
    height, width, channels = image.shape
    output = np.zeros_like(image)

    # 计算滤波器的半径
    radius = kernel_size // 2

    for c in range(channels):  # 针对每个颜色通道进行处理
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                # 提取滤波器覆盖的区域
                neighbors = image[i - radius:i + radius + 1, j - radius:j + radius + 1, c]

                # 计算邻域的中值作为当前像素的值
                output[i, j, c] = np.median(neighbors)

    return output

# 读取彩色图像
file_to_open = 'work01.jpg'
img = cv2.imread(file_to_open)

# 应用中值滤波
denoised_image = median_filter_color(img, kernel_size=5)  # 5表示核的大小，可以根据需要调整

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
