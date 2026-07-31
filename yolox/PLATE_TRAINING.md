# Training the plate DETECTOR (pure C++ YOLOX)

The detector is a YOLOX-tiny (8 classes: `car, small-truck, truck, bus, motorbike, bicycle, person,
plate`). `pure/train_cli.cpp` is a full training loop — dataset → SimOTA → YOLOX loss (IoU+obj+cls)
→ Adam (warmup+cosine) → per-epoch val mAP → save `last.pt`/`best.pt` — no Python at run time.

## 1. Make an 8-class init (COCO backbone + fresh nc-class head)
```sh
pip install torch loguru tabulate thop        # once (torch.hub loads Megvii YOLOX)
python pure/ref/export_plate_init.py 416 8 yolox_tiny     # -> pure/ref/data_unf_plate/
```
This loads the pretrained COCO yolox-tiny, replaces the 3 classification-head convs with `nc=8`
(random init), and dumps unfused conv+BN weights + `manifest_unfused.txt` + `names.txt` + `io.txt`.

## 2. Dataset (standard Ultralytics YOLO format)
```
mydata/train/images/*.jpg      mydata/train/labels/*.txt   # same basename
mydata/val/images/*.jpg        mydata/val/labels/*.txt
```
Each label line: `<cls> <xc> <yc> <w> <h>` normalized to [0,1]. For plates use class `7`; include the
vehicle classes (0–6) too if you have them. (`build/synth_plate/` is a synthetic example the repo can
generate for a smoke test.)

## 3. Train
```sh
cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_cli.cpp
#         train_dir                 val_dir                 epochs batch init imgsz mosaic nc initdir
train_cli mydata/train/images  mydata/val/images  100 8 "" 416 1 8 pure/ref/data_unf_plate/
```
`nc` (arg 8) and the init dir (arg 9) are the plate-specific additions. Verified end to end on the
synthetic set: **loss 12.5 → 5.4** in 3 epochs (mAP needs real data + more epochs). Output `best.pt`
is a standard YOLOX state_dict.

## 4. Use the trained detector
Export `best.pt` back to ONNX (megvii YOLOX export) and drop it in as `plate_yolox_tiny.onnx`, or run
it directly through the pure-C++ net. The browser demo (`../wasm/plate.html`) then detects with your
model. Note: the shipped `plate_yolox_tiny.onnx` (the pipeline's model) is a **ReLU** variant; the
training path here uses standard **SiLU** YOLOX — retrain fully (don't mix the ReLU weights).
