import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian
import cv2


# 得到卷积高斯模糊核PSF
def gaussian_kernel(kernel_size):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


# 给清晰图片加上卷积模糊核，给图片制造高斯模糊
def apply_blur(input_img, psf, epsilon):
    blurred = cv2.filter2D(input_img, -1, psf)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


# 维纳逆波实现图像去除模糊
'''
def apply_wiener(input_img, psf, epsilon, k=0.001):
    s = psf.shape
    input_fft = fft.fft2(input_img, s)
    psf_fft = fft.fft2(psf) + epsilon
    psf_fft_1 = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + k)
    result = fft.ifft2(input_fft * psf_fft_1)
    result = np.abs(fft.fftshift(result))
    return result
'''


def apply_wiener(input_img, psf, epsilon, k=0.001):
    # psf /= np.sum(psf)
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


# 对高斯模糊进行单一通道进行处理
def gaussian_main_process(input_image):
    gaussian_blur_channels = []

    img_height, img_width = input_image.shape[:2]
    # 高斯模糊模糊核
    gaussian_blur_psf = gaussian_kernel(15)

    gaussian_blurred_result = np.abs(apply_blur(input_image, gaussian_blur_psf, 1e-3))

    new_gaussian_blur_psf = gaussian_kernel(15)
    gaussian_wiener_result = apply_wiener(gaussian_blurred_result, new_gaussian_blur_psf, 1e-3)

    gaussian_blur_channels.append((clip_and_cast(gaussian_blurred_result), clip_and_cast(gaussian_wiener_result)))

    return gaussian_blur_channels


if __name__ == '__main__':
    input_image = cv2.imread('car.jpg')
    b_channel, g_channel, r_channel = cv2.split(input_image.copy())

    gaussian_final_result = []
    for channel in [b_channel, g_channel, r_channel]:
        processed_channel = gaussian_main_process(channel)
        gaussian_final_result.append(processed_channel)

    gaussian_blurred_img = cv2.merge(
        [gaussian_final_result[0][0][0], gaussian_final_result[1][0][0], gaussian_final_result[2][0][0]])

    wiener_deblurred_img = cv2.merge(
        [gaussian_final_result[0][0][1], gaussian_final_result[1][0][1], gaussian_final_result[2][0][1]])

    cv2.imwrite('Gaussian_Blurred_Image.JPG', gaussian_blurred_img)
    cv2.imwrite('Wiener_Deblurred_Image.JPG', wiener_deblurred_img)