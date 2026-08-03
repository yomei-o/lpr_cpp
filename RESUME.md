# RESUME — lpr_cpp remaining work

Japanese license-plate system = **detection (yolox/)** + **classification (pure/)**. Done so far:
forward parity (classifier 3.3e-05, detector 1.4e-05), classifier GPU (dnet_lpr, CPU-thrust 9/9 +
nvcc compiles), classifier training loop + real-data loader, detector training loop (NC=8), and the
**end-to-end browser demo** (two chained WASM modules). What's left:

## ⏭ Colab GPU verification (only the GPU *execution* is pending — code compiles + CPU-thrust-verified)
1. **LPR classifier device** — `colab/gpu_check_lpr.ipynb` on a T4 → expect `argmax 9/9 MATCH` +
   `backend: GPU (CUDA)`. (nvcc -DUSE_CUDA already compiles locally for sm_75.)
2. **YOLOX detector device** — `yolox/colab/` notebooks (came from yolox_cpp) on a T4.
3. (nice) build the WASM demos on Colab too, or just open the Pages links.

## ⏭ Real-data training (the repo ships the loops, not the private JP-plate data)
4. **Detector** — follow `yolox/PLATE_TRAINING.md`: `export_plate_init.py` → 8-class init, then
   `train_cli <train> <val> <epochs> <batch> "" 416 1 8 pure/ref/data_unf_plate/` on real plate images
   with YOLO-format labels (class 7 = plate). Synthetic smoke test gave loss 12.5→5.4; real data +
   more epochs is needed for real mAP. Export `best.pt` → ONNX → drop in as the detector.
5. **Classifier** — `train_lpr <steps> <batch> <lr> --data <dir> --save ckpt.bin` on real plate crops
   (`dir/labels.txt` = `<file> <9 indices>`; use `pure/ref/make_labels.py` to convert 品川/500/あ/1234
   annotations). Then copy `pure/ref/manifest.txt` next to the ckpt and rename to `weights.bin` for
   inference / the WASM demo.

## ✅ Speed / polish (DONE 2026-08-01)
6. ✅ **Eigen** CPU speedup — `-DUSE_EIGEN -Ipure/third_party/eigen_flat -arch:AVX2` → ~2× faster
   classifier training (40.7s → 20.0s / 6 steps), parity kept (3.28e-05).
7. ✅ **Device training** for the classifier — `dtrain_lpr.cpp` (device fwd train-mode + host-bridged
   9-head CE + DAdam). CPU-thrust runs; nvcc -DUSE_CUDA compiles (sm_75). GPU run → colab notebook.
8. ✅ **Detector WASM** — `-msimd128` → 7.8s → 3.3s/frame (button-press).
9. ✅ **Python-free** — shipped `yolox/pure/ref/data_unf_plate/` (8-class detector init) + classifier
   `weights.bin` + plate onnx, so train/infer/WASM need no Python (only one-time extraction scripts do).
10. ✅ **`lpr_infer`** — pure-C++ classifier inference on a real crop (image → 品川 371 ら 100), `/utf-8`.

## What the detector can and cannot do (measured 2026-08-03, don't re-derive this)
The shipped detector is trained on traffic-camera footage. Three things each cost it a lot, and a
hand-held plate in front of a webcam hits all three at once:

| condition | best score on the same photo |
|---|---|
| plate ≈ 5–12% of frame width, on a vehicle, white | **0.85** |
| same, yellow plate | 0.40 |
| same, **black plate (黒地に黄文字)** | **0.21** |
| plate ~46% of the frame, hand-held, no vehicle | **0.02, and in the wrong place** (floor 0.005) |

So "0 detections" on a hand-held black plate is the detector working as trained, not a bug. The fix
is not a lower threshold — it is **`plate.html`'s 「枠を手で指定」**: drag a box, skip the detector,
hand the crop straight to the classifier, which reads that same photo correctly. Retraining on
close-up / black plates is the real fix (§4).

## Classifier: the region head needs multi-crop (pure/lpr_tta.hpp)
地域名 is 133 classes decided by a few small glyphs, and it **flips with a few percent of crop
margin** — the same real photo read 奄美 / 横浜 / 練馬 depending on framing. `recognize_tta()` reads
6 margins (−6%…+10%) and **sums the per-head probabilities**; the summed winner's share is reported
as an agreement score, so a shaky region can be shown as shaky instead of asserted:

    single crop:  奄美 480 り 4567
    TTA:          横浜 480 り 4567    region 0.62  class 1.00/1.00/1.00  hira 0.98  num 1.00/1.00/1.00/0.98

**Do not softmax the heads again.** `net_lpr.hpp` already applies `softmax_rows` at inference and
each head sums to exactly 1.0; softmaxing a second time squashed the 133-class head to 0.01 against
a uniform 0.0076 and the vote became noise. `accumulate_probs()` asserts the sum ≈ 1 to keep that
from coming back. Same class of bug as the detector's double sigmoid.

- Native: `lpr_infer <weights_dir> <frame.png> --box x0 y0 x1 y1` (add `--single` for one crop).
- WASM: `fn_recognize_box(rgba, W, H, x0,y0,x1,y1)` + `fn_conf()`; checked by
  `node wasm/test_lpr_box.js <frame.rgba> <w> <h> <box…>` — it reproduced the native reading exactly.

## Notes / gotchas
- **Two engines**: `pure/` (LPR, facenet-derived, has dtensor GPU) and `yolox/` (detector, own engine)
  have clashing global symbols → cannot co-link → the end-to-end demo is **two WASM modules**.
- The shipped `yolox/plate_yolox_tiny.onnx` is a **ReLU** yolox-tiny; `train_cli` uses standard
  **SiLU** YOLOX — retrain fully from real data, don't mix the ReLU weights.
- Plate = class 7 of `car,small-truck,truck,bus,motorbike,bicycle,person,plate`.
- Live demos: https://yomei-o.github.io/lpr_cpp/wasm/plate.html (detect+recognize) and `/wasm/`
  (classifier only). nvcc build env + emsdk paths: see the facenet_cpp memory.
