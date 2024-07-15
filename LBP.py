import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# 读取灰度图像
gray_image = cv2.imread('gray_image1.jpg', cv2.IMREAD_GRAYSCALE)

# 定义 LBP 的参数
radius = 1
n_points = 8 * radius

# 计算 LBP 特征
lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')

# 计算 LBP 特征直方图
n_bins = int(lbp_image.max() + 1)
hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))

# 显示 LBP 特征直方图
import matplotlib.pyplot as plt

plt.bar(range(n_bins), hist)
plt.title('LBP Texture Feature Histogram')
plt.xlabel('LBP Code')
plt.ylabel('Frequency')
plt.show()
