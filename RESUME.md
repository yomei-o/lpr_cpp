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

## ⏭ Speed / polish
6. **Eigen** CPU speedup for the classifier training (`-DUSE_EIGEN -Ipure/third_party/eigen_flat`,
   conv/matmul already route through the `bk::` seam) — classifier train is ~15 s/step on plain CPU.
7. **Device training** for the classifier (only device *eval* forward exists via `dnet_lpr`; add a
   device train path like facenet's dtrain if GPU training is wanted).
8. **Detector WASM speed** — plate detect is ~3.3 s/frame (SIMD). Options: run at a smaller input,
   or a lighter model; currently button-press, not live.

## Notes / gotchas
- **Two engines**: `pure/` (LPR, facenet-derived, has dtensor GPU) and `yolox/` (detector, own engine)
  have clashing global symbols → cannot co-link → the end-to-end demo is **two WASM modules**.
- The shipped `yolox/plate_yolox_tiny.onnx` is a **ReLU** yolox-tiny; `train_cli` uses standard
  **SiLU** YOLOX — retrain fully from real data, don't mix the ReLU weights.
- Plate = class 7 of `car,small-truck,truck,bus,motorbike,bicycle,person,plate`.
- Live demos: https://yomei-o.github.io/lpr_cpp/wasm/plate.html (detect+recognize) and `/wasm/`
  (classifier only). nvcc build env + emsdk paths: see the facenet_cpp memory.
