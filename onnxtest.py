import onnxruntime as ort 

model_path = r"runs\train\exp2\weights\best.onnx" 

sess = ort.InferenceSession(model_path) 

print(sess.get_inputs()[0].name) 

print(sess.get_inputs()[0].shape) 

 