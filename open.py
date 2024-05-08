import cv2
import numpy as np
from numpy.fft import fft2, ifft2

# 维纳逆波实现图像去除模糊
def apply_wiener(input_img, psf, epsilon, k=0.001):
    dummy = np.copy(input_img)
    dummy = fft2(dummy)
    psf = fft2(psf, s=input_img.shape)
    psf = np.conj(psf) / (np.abs(psf) ** 2 + k)
    dummy = dummy * psf
    dummy = np.abs(ifft2(dummy))
    return dummy

# 数组裁切
def clip_and_cast(array):
    array = np.where(array < 0, 0, array)
    array = np.where(array > 255, 255, array)
    array = array.astype(np.uint8)
    return array

# 局部Wiener去模糊
def local_wiener_deblur(input_img, psf, mask, epsilon, k=0.001):
    wiener_deblurred_result = np.copy(input_img)
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            if mask[i, j] == 1:  # 只在模糊区域进行去模糊处理
                local_patch = input_img[max(0, i - 2):min(input_img.shape[0], i + 3),
                                       max(0, j - 2):min(input_img.shape[1], j + 3)]
                local_psf = psf[max(0, 2 - i):5 - max(0, i + 3 - input_img.shape[0]),
                                max(0, 2 - j):5 - max(0, j + 3 - input_img.shape[1])]
                local_deblurred_patch = apply_wiener(local_patch, local_psf, epsilon, k)
                wiener_deblurred_result[i, j] = local_deblurred_patch[2, 2]  # 取中心像素作为处理结果
    return clip_and_cast(wiener_deblurred_result)

def specify_blur_region(image):
    # 复制图像以免修改原始图像
    img_copy = image.copy()

    # 显示图像并允许用户绘制矩形
    cv2.imshow('Specify Blur Region', img_copy)
    rect = cv2.selectROI('Specify Blur Region', img_copy, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow('Specify Blur Region')

    # 返回矩形坐标
    return rect

if __name__ == '__main__':
    # 读取图像
    input_image = cv2.imread('car.png')  # 读取灰度图像

    # 获取模糊区域的边界框坐标
    blur_region = specify_blur_region(input_image)

    # 提取模糊区域的坐标信息
    x, y, w, h = blur_region

    # 创建模糊区域的掩码
    blur_mask = np.zeros_like(input_image)
    blur_mask[y:y+h, x:x+w] = 1

    # 使用一个单位矩阵作为 PSF
    psf = np.eye(5) / 5  # 5x5的单位矩阵

    # 进行局部Wiener去模糊处理
    deblurred_img = local_wiener_deblur(input_image, psf, blur_mask, epsilon=1e-3)

    # 显示去模糊结果
    cv2.imshow('Deblurred Image', deblurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()