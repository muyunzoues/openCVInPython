import cv2
import numpy as np

def erosion(image, kernel):
    rows, cols = image.shape
    padded_image = np.pad(image, ((3, 3), (3, 3)), mode='constant')
    eroded_image = np.zeros_like(image)

    for i in range(3, rows + 3):
        for j in range(3, cols + 3):
            neighborhood = padded_image[i - 3:i + 4, j - 3:j + 4]  # 调整邻域数组的大小以匹配核的大小
            # 使用整个核的大小进行计算
            if np.min(neighborhood + kernel) == 255 * (kernel.shape[0] * kernel.shape[1]):
                eroded_image[i - 3, j - 3] = 255

    return eroded_image




def dilation(image, kernel):
    rows, cols = image.shape
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    dilated_image = np.zeros_like(image)

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            neighborhood = padded_image[i - 1:i + 2, j - 1:j + 2]
            # 将核的大小调整为与邻域相匹配
            adjusted_kernel = np.ones_like(neighborhood) * kernel[1, 1]
            # 使用逐元素的最大值计算
            if np.max(neighborhood + adjusted_kernel) >= 255:
                dilated_image[i - 1, j - 1] = 255

    return dilated_image



def opening(image, kernel):
    eroded_image = erosion(image, kernel)
    opened_image = dilation(eroded_image, kernel)
    return opened_image


def closing(image, kernel):
    dilated_image = dilation(image, kernel)
    closed_image = erosion(dilated_image, kernel)
    return closed_image


# 读取图像
img = cv2.imread('dentalXray-salt-noise.tif', 0)
# 应用高斯模糊
blurred = cv2.GaussianBlur(img, (7, 7), 0)
# 定义形态学核
kernel = np.ones((7, 7), np.uint8)
# 使用开运算消除噪声
opening_result = opening(blurred, kernel)
# 使用闭运算连接断开的部分
closing_result = closing(opening_result, kernel)
# 保存处理后的图像
cv2.imwrite('processed_image.tif', closing_result)
# 显示处理后的图像
cv2.imshow('Result', closing_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
