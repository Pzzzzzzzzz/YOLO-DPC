import cv2

# 读取彩色图像
image = cv2.imread(r'color_datasets\images\train\2022-08-24_211402.jpg')

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 显示灰度图像
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)

# 保存灰度图像
cv2.imwrite('gray_image1.jpg', gray_image)

# 关闭窗口
cv2.destroyAllWindows()