import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 文件夹路径
folder_path_1 = r'histogram-data/1'
folder_path_2 = r'histogram-data/2'

# 加载图像数据并计算直方图
def calculate_histogram(folder_path):
    hist_list = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_list.append(hist)
    return hist_list

# 分别计算两个类别的直方图
hist_list_1 = calculate_histogram(folder_path_1)
hist_list_2 = calculate_histogram(folder_path_2)

# 绘制直方图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for hist in hist_list_1:
    ax1.plot(hist)
ax1.set_title('Histogram 1')

for hist in hist_list_2:
    ax2.plot(hist)
ax2.set_title('Histogram 2')

plt.tight_layout()
plt.show()
