import collections
import numpy as np
import tflite as tfl_pkg
from tflite import TensorType, BuiltinOperator as BO

print("TensorType enum:", {k: v for k, v in vars(TensorType).items() if isinstance(v, int)})

base = "/home/joshua/work/test_yolov8_ultralytics"
buf = open(f"{base}/tflite/yolov8n_int8.tflite", "rb").read()
mdl = tfl_pkg.Model.GetRootAs(buf, 0)
g = mdl.Subgraphs(0)

codes = []
for i in range(mdl.OperatorCodesLength()):
    oc = mdl.OperatorCodes(i)
    try:
        codes.append(oc.BuiltinCodeAsIndex())
    except Exception:
        codes.append(oc.BuiltinCode())
code2name = {int(v): k for k, v in vars(BO).items() if isinstance(v, int)}
EN = {v: k for k, v in vars(TensorType).items() if isinstance(v, int)}

def tinfo(idx):
    t = g.Tensors(int(idx))
    q = t.Quantization()
    ns = (q.ScaleLength() if hasattr(q, "ScaleLength") else q.ScalesLength()) if q else 0
    sc = None
    if ns:
        get = q.Scale if hasattr(q, "Scale") else q.Scales
        sc = (float(get(0)), ns, q.QuantizedDimension())
    return idx, t.Name().decode(), EN.get(t.Type(), str(t.Type())), sc

tcnt = collections.Counter()
for ti in range(g.TensorsLength()):
    _, _, dt, sc = tinfo(ti)
    tcnt[(dt, "quantized" if sc else "plain")] += 1
print("tensor dtypes:", dict(tcnt))

ops = []
producer = {}
for oi in range(g.OperatorsLength()):
    op = g.Operators(oi)
    nm = code2name.get(codes[op.OpcodeIndex()], "?")
    ins = [tinfo(op.Inputs(j)) for j in range(op.InputsLength())]
    outs = [tinfo(op.Outputs(j)) for j in range(op.OutputsLength())]
    ops.append((nm, ins, outs))
    for o in outs:
        producer[o[0]] = oi

wdt = collections.Counter()
act_in = collections.Counter()
for nm, ins, _ in ops:
    if nm == "CONV_2D":
        _, _, dt, sc = ins[1]
        wdt[(dt, "per-channel" if sc and sc[1] > 1 else ("per-tensor" if sc else "plain"))] += 1
        _, _, dt2, sc2 = ins[0]
        act_in[(dt2, bool(sc2))] += 1
print("CONV_2D weights:", dict(wdt))
print("CONV_2D activation input:", dict(act_in))

out_idx = g.Outputs(0)
name, dt, sc = tinfo(out_idx)[1], tinfo(out_idx)[2], tinfo(out_idx)[3]
print(f"\ngraph output: {name} dtype={dt} quant={sc}")
cur = out_idx
for depth in range(6):
    if cur not in producer:
        break
    nm, ins, outs = ops[producer[cur]]
    def fmt(i):
        if i[3]:
            tail = f"x{i[3][1]}ch]" if i[3][1] > 1 else "]"
            return f"{i[2]}[q {i[3][0]:.5f}{tail}"
        return f"{i[2]}[plain]"
    print(f"  [{depth}] {nm}: out={outs[0][2]}{'(q)' if outs[0][3] else '(plain)'}  in=" + ", ".join(fmt(i) for i in ins[:6]))
    nxt = [i for i in ins if i[0] in producer]
    cur = nxt[0][0] if nxt else None

# scales on the box path vs conf path at the head (LOGISTIC out = conf)
logistic_out_scales = []
mul_out_scales = []
for nm, ins, outs in ops:
    if nm == "LOGISTIC" and outs[0][3] and outs[0][3][0] < 0.02:
        logistic_out_scales.append(outs[0][3][0])
    if nm == "MUL" and outs[0][3] and outs[0][3][0] > 0.5:
        mul_out_scales.append(outs[0][3])
print(f"\nconf LOGISTIC out scales (n={len(logistic_out_scales)}): min={min(logistic_out_scales):.5f} max={max(logistic_out_scales):.5f}")
print(f"box MUL out scales (n={len(mul_out_scales)}): sample={[round(s[0],4) for s in mul_out_scales[:6]]}")

# functional: quantization grid of conf vs box in real output
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path=f"{base}/tflite/yolov8n_int8.tflite")
it.allocate_tensors()
inp = it.get_input_details()[0]
img = np.zeros((1, 640, 640, 3), dtype=np.float32)
it.set_tensor(tensor_index=inp["index"], value=img)
it.invoke()
o = it.get_tensor(it.get_output_details()[0]["index"])[0]  # [84, 8400]
conf = np.sort(np.unique(o[4:, :].ravel()))
step = np.median(np.diff(np.unique(np.round(conf, 4))))
box = np.sort(np.unique(o[:4, :].ravel()))
bstep = np.min(np.diff(np.unique(np.round(box, 2)))[np.diff(np.unique(np.round(box, 2))) > 1e-4])
print(f"\nRUNTIME grid: conf unique={len(conf)} typical step~{step:.5f} | box unique={len(box)} step~{bstep:.4f}")
print(f"conf value examples: {conf[-6:]}")
