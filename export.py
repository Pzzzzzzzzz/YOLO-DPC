from ultralytics import YOLO

# Load a model

model = YOLO(r'runs\train\2\weights\best.pt')  # load a custom trained

# Export the model
model.export(format='onnx')