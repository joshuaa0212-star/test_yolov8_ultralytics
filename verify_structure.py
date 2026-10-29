import json
import onnx
import numpy as np

# --- 1) ONNX structure: Q/DQ node counts, weight dtypes, IO types ---
m = onnx.load("/home/joshua/work/test_yolov8_ultralytics/onnx/yolov8n_int8.onnx")
opmap = {}
for n in m.graph.node:
    opmap[n.op_type] = opmap.get(n.op_type, 0) + 1
qdq = {k: v for k, v in opmap.items() if k in ("QuantizeLinear", "DequantizeLinear")}
weights = [i.data_type for i in m.graph.initializer]
from onnx import TensorProto
int8w = sum(1 for t in weights if t == TensorProto.INT8)
uint8w = sum(1 for t in weights if t == TensorProto.UINT8)
io = []
for t in list(m.graph.input) + list(m.graph.output):
    et = t.type.tensor_type.elem_type
    shape = [d.dim_value if d.HasField("dim_value") else "?" for d in t.type.tensor_type.shape.dim]
    dt = {1: "FLOAT", 2: "UINT8", 3: "INT8", 6: "INT32", 7: "INT64", 10: "FLOAT16"}.get(et, f"type{et}")
    io.append((t.name, dt, shape))
print("ONNX QDQ nodes:", qdq)
print(f"ONNX initializers: total={len(weights)} int8={int8w} uint8={uint8w}")
print("ONNX IO:", io)

# --- 2) TFLite: interpreter tensor details ---
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path="/home/joshua/work/test_yolov8_ultralytics/tflite/yolov8n_int8.tflite")
it.allocate_tensors()
def details(lst):
    out = []
    for t in lst:
        q = t.get("quantization_parameters", {})
        scales = q.get("scales", [])
        out.append((t["name"], str(t["dtype"]), "q_scale_n=" + str(len(scales)) if len(scales) else "float"))
    return out
print("TFLITE in:", details(it.get_input_details()))
print("TFLITE out:", details(it.get_output_details()))
