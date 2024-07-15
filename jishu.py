
from ultralytics import YOLO
import cv2
import os

def draw_transparent_background(image, x, y, width, height, bg_color):
    """ 在图像上绘制半透明背景矩形 """
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), bg_color, -1)  # 在图像上绘制矩形
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)  # 将矩形融合到原始图像上以增加透明度

if __name__ == "__main__":
    # 加载YOLOv8n模型
    model = YOLO(r"yolov8n.pt")

    # 使用模型对图像进行预测
    results = model.predict(
        source=r"ultralytics\assets\bus.jpg",  # 指定待预测的图像路径
        save=True,  # 保存预测结果的图像
        imgsz=640,  # 设置图像大小
        conf=0.25,  # 设置目标检测的置信度阈值
        iou=0.45,  # 设置非极大值抑制的交并比阈值
        show=False,  # 设置是否显示预测结果
        classes=[0], # 要计数的类别
        
        # ...[省略其它参数]
    )

    counts = {}  # 用于存储每个类别的计数
    for result in results:
        boxes = result.boxes.cpu().numpy()  # 获取检测到的边界框
        original_image_path = result.path  # 获取原始图像的路径
        save_dir = result.save_dir  # 获取保存处理后图像的目录

        # 对每个检测到的对象进行计数
        for box in boxes:
            cls = int(box.cls[0])  # 获取类别索引
            if cls not in counts:
                counts[cls] = 1
            else:
                counts[cls] += 1

    # 从原始图片路径中提取文件名
    image_filename = os.path.basename(original_image_path)

    # 构建处理后图像的完整路径
    output_image_path = os.path.join(save_dir, image_filename)
    img = cv2.imread(output_image_path)  # 读取处理后的图像

    # 设置文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)  # 文本颜色
    bg_color = (0, 0, 0)  # 背景颜色

    # 计算文本区域的总高度
    text_height_total = 0
    margin = 5  # 设置边距
    for key in counts.keys():
        text = f"{model.names[key]} : {counts[key]}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_height_total += text_height + margin * 2

    # 绘制包含所有文本的背景矩形
    bg_x, bg_y = 10, 25
    bg_width = max([cv2.getTextSize(f"{model.names[key]} : {counts[key]}", font, font_scale, font_thickness)[0][0] for key in counts.keys()]) + margin * 2
    bg_height = text_height_total
    draw_transparent_background(img, bg_x, bg_y, bg_width, bg_height, bg_color)

    # 在背景矩形上逐行添加文本
    y0 = bg_y + margin + 7
    for key in counts.keys():
        text = f"{model.names[key]} : {counts[key]}"
        y = y0
        cv2.putText(img, text, (bg_x + margin, y), font, font_scale, text_color, font_thickness)
        y0 += text_height + margin * 2

    # 将带有计数信息的图像保存到文件
    cv2.imwrite(f'{image_filename}_counts_on_image.jpg', img)

