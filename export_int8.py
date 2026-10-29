from ultralytics import YOLO

model = YOLO("yolov8n.pt")

data = "/home/joshua/work/test_yolov8_ultralytics/datasets/coco8.yaml"

f_onnx = model.export(format="onnx", quantize=8, data=data, imgsz=640)
print("ONNX_OUT:", f_onnx)

f_tflite = model.export(format="litert", quantize=8, data=data, imgsz=640)
print("TFLITE_OUT:", f_tflite)
