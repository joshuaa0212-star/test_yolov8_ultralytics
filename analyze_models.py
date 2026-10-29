import collections, io, json, zipfile
import numpy as np
import onnx

base = "/home/joshua/work/test_yolov8_ultralytics"

# ---------- 1) PT ----------
print("=" * 20, "PT", "=" * 20)
import torch
from ultralytics import YOLO
y = YOLO(f"{base}/yolov8n.pt")
mm = y.model
nparams = sum(p.numel() for p in mm.parameters())
dtypes = collections.Counter(str(p.dtype) for p in mm.parameters())
mods = collections.Counter(type(m).__name__ for m in mm.modules())
print(f"params={nparams:,} dtypes={dict(dtypes)}")
print(f"modules Conv={mods['Conv']} SiLU={mods['SiLU']} C2f={mods['C2f']} SPPF={mods['SPPF']} Detect={mods['Detect']}")
det = mm.model[-1]
nbox = sum(len(s) for s in det.cv2)
ncls = sum(len(s) for s in getattr(det, "cv3", det.cv2))
dflv = getattr(det.dfl, "reg_max", getattr(det.dfl, "n", "?")) if hasattr(det, "dfl") else "?"
print(f"Detect: nl={det.nl}, box convs={nbox}, cls convs={ncls}, strides={det.stride}, DFL reg_max={dflv}")
print(f"pt file bytes={__import__('os').path.getsize(base + '/yolov8n.pt'):,}")

# ---------- 2) ONNX INT8 ----------
print("=" * 20, "ONNX", "=" * 20)
m = onnx.load(f"{base}/onnx/yolov8n_int8.onnx")
g = m.graph
ops = collections.Counter(n.op_type for n in g.node)
print("op counts:", dict(ops))
# per-node producers
prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n
# Conv coverage: Conv consuming a Q node output (activation quantized)
convs = [n for n in g.node if n.op_type == "Conv"]
qc = sum(1 for n in convs if prod.get(n.input[0]) and prod[n.input[0]].op_type == "QuantizeLinear")
print(f"Conv total={len(convs)}, with Q on activation input={qc}")
# head nodes: what feeds Sigmoid/Softmax/Concat
for tname in ("Sigmoid", "Softmax", "Concat"):
    for n in [x for x in g.node if x.op_type == tname][:3]:
        src = prod.get(n.input[0])
        print(f"{tname} <- {src.op_type if src else 'graph input'}")
# output0 chain
o0 = [v for v in g.output][0]
node = prod[o0.name]
ins = [(i, prod[i].op_type if i in prod else "input") for i in node.input]
print(f"output0 fed by {node.op_type}, inputs={ins}")
# activation zero-points: signed vs unsigned
init = {i.name: i for i in g.initializer}
zt = set()
for n in g.node:
    if n.op_type == "QuantizeLinear" and len(n.input) > 2:
        zp = init.get(n.input[2])
        if zp is not None:
            arr = onnx.numpy_helper.to_array(zp)
            zt.add((str(arr.dtype), float(arr.reshape(-1)[0])))
print("activation Q zero_point (dtype, value) samples:", sorted(zt)[:4])
# weight quantization axis (per-channel?)
axes = set()
for n in g.node:
    if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
        for a in n.attribute:
            if a.name == "axis":
                axes.add((n.op_type, a.i))
print(f"Q/DQ axis attributes (per-channel): {sorted(axes) if axes else 'none -> per-tensor only'}")
dq = sum(1 for n in convs if prod.get(n.input[0]) and prod[n.input[0]].op_type == "DequantizeLinear" and prod[prod[n.input[0]].input[0]].op_type == "QuantizeLinear")
print(f"Conv with Q->DQ on activation input: {dq}/{len(convs)}")

# ---------- 3) TFLITE ----------
print("=" * 20, "TFLITE", "=" * 20)
import tflite

def ns(q):
    if q is None:
        return 0
    for f in ("ScalesLength", "ScaleLength"):
        if hasattr(q, f):
            return getattr(q, f)()
    return 0

buf = open(f"{base}/tflite/yolov8n_int8.tflite", "rb").read()
mdl = tflite.Model.GetRootAs(buf, 0)
print(f"version={mdl.Version()} subgraphs={mdl.SubgraphsLength()}")
TT = {0: "BOOL", 1: "UINT8", 2: "INT64", 3: "INT8", 4: "UINT16", 6: "UINT32", 8: "STRING", 9: "FLOAT32", 10: "INT64b", 11: "INT32", 12: "UINT64", 13: "INT16", 16: "FLOAT16", 17: "UINT4", 18: "BOOLx"}
codes = []
for i in range(mdl.OperatorCodesLength()):
    oc = mdl.OperatorCodes(i)
    try:
        codes.append(oc.BuiltinCodeAsIndex())
    except Exception:
        codes.append(oc.BuiltinCode())
import tflite.BuiltinOperator as BO
code2name = {int(v): k for k, v in vars(BO).items() if isinstance(v, int)}
for si in range(mdl.SubgraphsLength()):
    g2 = mdl.Subgraphs(si)
    print(f"-- subgraph {si}: ops={g2.OperatorsLength()} tensors={g2.TensorsLength()}")
    tcnt = collections.Counter()
    qinfo = collections.Counter()
    for ti in range(g2.TensorsLength()):
        t = g2.Tensors(ti)
        tt = TT.get(t.Type(), str(t.Type()))
        tcnt[tt] += 1
        q = t.Quantization()
        if ns(q) > 0:
            qinfo[(tt, "per-channel" if ns(q) > 1 else "per-tensor")] += 1
    print("tensor dtypes:", dict(tcnt))
    print("quantized tensors:", dict(qinfo))
    opc = collections.Counter()
    floatops = collections.Counter()
    interesting = []
    for oi in range(g2.OperatorsLength()):
        op = g2.Operators(oi)
        nm = code2name.get(codes[op.OpcodeIndex()], "?")
        opc[nm] += 1
        io_t = []
        for idx in (op.Inputs(j) for j in range(op.InputsLength())):
            t = g2.Tensors(int(idx))
            io_t.append((TT.get(t.Type(), "?"), "Q" if (ns(t.Quantization()) > 0) else "-"))
        outs = []
        for idx in (op.Outputs(j) for j in range(op.OutputsLength())):
            t = g2.Tensors(int(idx))
            outs.append((TT.get(t.Type(), "?"), "Q" if (ns(t.Quantization()) > 0) else "-"))
        allfloat = all(qf == "-" for _, qf in io_t + outs)
        if allfloat:
            floatops[nm] += 1
        if nm in ("LOGISTIC", "SOFTMAX", "CONCATENATION", "QUANTIZE", "DEQUANTIZE", "RESCALE", "SOFTMAX_XY"):
            interesting.append((nm, io_t, outs))
    print("op counts:", dict(opc))
    print("fully-FLOAT32 ops:", dict(floatops))
    for nm, ins_, outs_ in interesting[:14]:
        print(f"  {nm}: in={ins_} out={outs_}")
    gi = [g2.Inputs(j) for j in range(g2.InputsLength())]; go = [g2.Outputs(j) for j in range(g2.OutputsLength())]; print("input:", gi, "output:", go)
    for io_idx in gi + go:
        t = g2.Tensors(int(io_idx))
        q = t.Quantization()
        print(f"  IO {t.Name().decode()}: type={TT.get(t.Type())} quant={'yes' if q and ns(q) else 'no'}")
# appended metadata zip
try:
    with zipfile.ZipFile(f"{base}/tflite/yolov8n_int8.tflite") as zf:
        md = json.loads(zf.read("metadata.json"))
    print("metadata.json:", json.dumps(md, ensure_ascii=False)[:600])
except Exception as e:
    print("metadata read fail:", e)
