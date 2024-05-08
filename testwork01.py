import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['FangSong']  # 设置字体以便正确显示汉字
plt.rcParams['axes.unicode_minus'] = False  # 正确显示连字符


MEDTH_ID=5


# 计算图片清晰度
def getImageVar(img):
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
    # 对图片用 3x3 拉普拉斯算子做卷积得到边缘  计算出方差，并最后返回。
    # 函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以建立的图像位数不够，会有截断。因此要使用64位有符号的数据类型，即 cv2.CV_64F。
    # 再用var函数求方差
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar


# 转接器
def handle(idx, img):
    if idx == 1: return handle_specific(img, linear) # 线性变化
    if idx == 2: return handle_specific(img, linear_up) # 分段线性变化
    if idx == 3: return handle_specific(img, Logarithmic) # 对数变换
    if idx == 4: return handle_specific(img, power) # 幂指变换
    if idx == 5: return handle_specific(img, cv2.equalizeHist)  # 直方图均衡化
    if idx == 6: return handle_specific(img, auto_equalizeHist)  # 自适应直方图均衡化
    if idx == 7: return handle_specific(img, laplacian)  # laplacian算子图像锐化
    if idx == 8: return handle_specific(img, non_sharpening)  # 非锐化掩蔽


# 处理函数
def handle_specific(img, func):
    img_list = [func(i) for i in cv2.split(img)]
    result = cv2.merge((img_list[0], img_list[1], img_list[2]))
    return result


# 线性变化
def linear(img):
    a, b = 1.5, 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] * a + b > 255:
                img[i][j] = 255
            else:
                img[i][j] = img[i][j] * a + b
    return img

# 分段线性变换-线性对比度拉伸,增强感兴趣区域
def linear_up(img):
    # 灰度值的最大最小值
    r_min, r_max = 255, 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > r_max:
                r_max = img[i, j]
            if img[i, j] < r_min:
                r_min = img[i, j]
    r1, s1 = r_min, 0
    r2, s2 = r_max, 255

    k = (s2 - s1) / (r2 - r1)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if r1 <= img[i, j] <= r2:
                img[i, j] = k * (img[i, j] - r1)
    return img

# 对数变换
def Logarithmic(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = math.log(1+img[i][j])
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.convertScaleAbs(img)
    return img

# 对数变换
def power(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = math.pow(img[i][j],1.2)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.convertScaleAbs(img)
    return img

# 自适应的直方图均衡化-非线性的对比度拉伸,增强感兴趣区域
def auto_equalizeHist(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


def laplacian(img):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # laplacian卷积核的一个模板
    lapkernel_img = cv2.filter2D(img, -1, kernel) # 做卷积
    img = img - lapkernel_img
    return img

def non_sharpening(img):
    blur_img = cv2.blur(img, (5, 5))
    mask_img = img - blur_img
    img = img + mask_img
    return img

img = cv2.imread(filename='work01.jpg', flags=1)
result = handle(MEDTH_ID, img)

print('原图的清晰度：', getImageVar(img))
print('处理之后的清晰度', getImageVar(result))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(img)
axes[0].set_title("原图")

axes[1].imshow(result)
axes[1].set_title("处理之后的图片")
plt.show()


