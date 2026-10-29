import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

base = "/home/joshua/work/test_yolov8_ultralytics"
models = {
    "FP32 (pt)": f"{base}/yolov8n.pt",
    "ONNX INT8": f"{base}/onnx/yolov8n_int8.onnx",
    "TFLite INT8": f"{base}/tflite/yolov8n_int8.tflite",
}
imgs = ["test1.jpg", "test2.jpg"]

# 1) predict + save annotated frames per model
outs = {}
for label, path in models.items():
    m = YOLO(path)
    name = label.replace(" ", "_").replace("(", "").replace(")", "")
    for img in imgs:
        src = f"{base}/test_images/{img}"
        m.predict(src, conf=0.25, verbose=False, save=True, save_conf=True,
                  project=f"{base}/results/raw", name=name)
        outs[(label, img)] = f"{base}/results/raw/{name}/{img}"

# 2) side-by-side comparison grid per test image (left -> right)
try:
    font = ImageFont.load_default(size=28)
except TypeError:
    font = ImageFont.load_default()

os.makedirs(f"{base}/results/visual", exist_ok=True)
for img in imgs:
    tiles = []
    for label in models:
        im = Image.open(outs[(label, img)]).convert("RGB")
        im.thumbnail((640, 640))
        tiles.append((label, im))
    W = sum(t.width for _, t in tiles)
    H = max(t.height for _, t in tiles) + 40
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    x = 0
    for label, t in tiles:
        grid.paste(t, (x, 40))
        d.text((x + 8, 8), label, fill="black", font=font)
        x += t.width
    outp = f"{base}/results/visual/{os.path.splitext(img)[0]}_compare.jpg"
    grid.save(outp, quality=90)
    print("SAVED:", outp)
