import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'ultralytics\cfg\models\v8\yolov8-Vanillanet.yaml')
    # model.load('yolov8n.pt') 
    model.train(data=r'color_datasets\data.yaml',
                
                cache=False,
                imgsz=640,
                epochs=500,
                single_cls=False,  # 是否是单类别检测
                batch=8,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='Adam', # using SGD
                # resume='', # 如过想续训就设置last.pt的地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                )