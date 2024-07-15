from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
path=r'runs\train\2\weights\best.onnx'
model = onnx.load(path)
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))

# 模型量化
yolov8_uint8 = 'yolov8_uint8.onnx'
quantize_dynamic(path, yolov8_uint8, weight_type=QuantType.QUInt8)

# 检查量化模型
model = onnx.load(yolov8_uint8)
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))


import os
onnx_model_path = yolov8_uint8
parameter_size = os.path.getsize(onnx_model_path) / (1024 * 1024)  # 将字节转换为MB
print(f"模型的参数大小：{parameter_size} MB")


# 保存量化模型
# quantized_model_path = 'E:\shibie\yolo\\uzi\c.oonx'
quantized_model_path = r'runs\train\2\weights\best.Dystatic.onnx'
onnx.save(model, quantized_model_path)
print(f"量化模型已保存至：{quantized_model_path}")