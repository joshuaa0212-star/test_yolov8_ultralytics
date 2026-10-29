from ultralytics import YOLO

base = "/home/joshua/work/test_yolov8_ultralytics"
models = {
    "fp32_pt": YOLO(f"{base}/yolov8n.pt"),
    "onnx_int8": YOLO(f"{base}/onnx/yolov8n_int8.onnx"),
    "tflite_int8": YOLO(f"{base}/tflite/yolov8n_int8.tflite"),
}

for name, m in models.items():
    r = m.predict(f"{base}/test_images/test1.jpg", conf=0.25, verbose=False)[0]
    tops = [
        (int(b.cls[0]), float(b.conf[0]))
        for b in sorted(r.boxes, key=lambda b: -float(b.conf[0]))[:5]
    ]
    print(f"PRED {name}: n={len(r.boxes)} top5={tops}")

for name, m in models.items():
    metrics = m.val(data=f"{base}/datasets/coco8.yaml", split="val", verbose=False, plots=False)
    print(f"VAL {name}: mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")
