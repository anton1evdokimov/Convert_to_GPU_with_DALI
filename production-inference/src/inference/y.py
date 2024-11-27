import sys
import onnx
filename = "yolov7.onnx"
model = onnx.load(filename)
print(onnx.checker.check_model(model))