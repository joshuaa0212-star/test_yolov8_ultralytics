import numpy as np
from PIL import Image

base = "/home/joshua/work/test_yolov8_ultralytics"
img = np.asarray(Image.open(f"{base}/test_images/test1.jpg").resize((640, 640)), dtype=np.float32) / 255.0
nchw = img.transpose(2, 0, 1)[None]  # (1,3,640,640)

def grid(a):
    u = np.unique(np.round(a, 6))
    d = np.diff(u)
    d = d[d > 1e-7]
    return len(u), (float(np.min(d)) if len(d) else 0.0)

from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=f"{base}/tflite/yolov8n_int8.tflite")
it.allocate_tensors()
ii = it.get_input_details()[0]
print("tflite input shape:", ii["shape"])
it.set_tensor(tensor_index=ii["index"], value=nchw.astype(ii["dtype"]))
it.invoke()
t_out = it.get_tensor(it.get_output_details()[0]["index"])[0]  # (84, 8400)
print("TFLITE out box grid:", grid(t_out[:4].ravel()), "conf grid:", grid(t_out[4:].ravel()))

import onnxruntime as ort
so = ort.SessionOptions()
sess = ort.InferenceSession(f"{base}/onnx/yolov8n_int8.onnx", so, providers=["CPUExecutionProvider"])
oinp = sess.get_inputs()[0]
print("onnx input shape:", oinp.shape)
o_out = sess.run(None, {oinp.name: np.ascontiguousarray(nchw, dtype=np.float32)})[0][0]
print("ONNX out box grid:", grid(o_out[:4].ravel()), "conf grid:", grid(o_out[4:].ravel()))
print("ONNX conf unique:", len(np.unique(o_out[4:].ravel())), "tflite conf unique:", len(np.unique(t_out[4:].ravel())))
print("conf max onnx=%.4f tflite=%.4f" % (o_out[4:].max(), t_out[4:].max()))
print("box range onnx=[%.2f,%.2f] tflite=[%.2f,%.2f]" % (o_out[:4].min(), o_out[:4].max(), t_out[:4].min(), t_out[:4].max()))
