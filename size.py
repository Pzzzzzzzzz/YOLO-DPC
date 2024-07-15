import os
onnx_model_path = r"runs\train\4\weights\best.pt"
parameter_size = os.path.getsize(onnx_model_path) / (1024 * 1024)  # 将字节转换为MB
print(f"模型的参数大小：{parameter_size} MB")
