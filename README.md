# lpr_cpp — pure C++ Japanese license-plate classifier (no PyTorch/TF, no CMake) — WIP

The license-plate **recognition** stage of the govtech traffic-count pipeline
(`Classificator/lpr`), ported to a dependency-free C++ autograd engine — **training and inference**,
no Python at run time. Plate **detection** is YOLOX (already ported in
[yolox_cpp](https://github.com/yomei-o/yolox_cpp)); this repo is the per-plate **classifier** that
reads the 地域名・分類番号・ひらがな・一連番号.

Reference model = the pipeline's `model_128x128_..._relu.onnx` (Keras/TF): a depthwise-separable
ResNet, 128 ch, that takes a 128×128 plate crop and emits **9 softmax heads** via two branches.
Full spec: [pure/ref/ARCH.md](pure/ref/ARCH.md).

## Browser demos (WebAssembly)
**▶ End-to-end (detect + recognize): https://yomei-o.github.io/lpr_cpp/wasm/plate.html** — camera →
YOLOX finds the plate → crop → LPR reads it. Two chained WASM modules (`yolox.wasm` detect +
`lpr.wasm` classify); detection runs the 83-layer ReLU yolox-tiny single-threaded (~seconds/frame,
so it's on button-press). Detector (`yolox/`) is parity-verified vs onnxruntime (worst 1.4e-05).

**▶ Classifier only (fit the plate in the box): https://yomei-o.github.io/lpr_cpp/wasm/** (GitHub Pages, HTTPS → camera works)

`wasm/` is a client-side app: aim the camera at a plate, fit it in the guide box, and it shows the
decoded **地域名・分類番号・ひらがな・一連番号**. The pure-C++ classifier compiled to WASM; the tracked
`pure/ref/weights.bin` (1.3 MB) is fetched at runtime — no server, no upload, no Emscripten needed:
```sh
python -m http.server 8000        # from the repo root
# open http://localhost:8000/wasm/  (camera needs localhost or HTTPS)
```
Prebuilt `wasm/lpr.js` + `wasm/lpr.wasm` are committed; node argmax matches the ONNX reference.
Detection isn't included — fill the guide box with the plate (YOLOX auto-crop is the sibling repo).

## Status — inference parity WORKING ✅
- **forward** (`net_lpr.hpp`): reproduces the ONNX — `pure/m1_lpr.cpp` = worst **3.3e-05** vs
  onnxruntime, argmax **9/9** on all heads.
- Architecture was extracted from the ONNX (no guessing) by `pure/ref/export_lpr.py` → `ref/manifest.txt`
  + `ref/weights.bin` (forward-order weights) + `ref/{input,refout}.bin` (parity).

```sh
python pure/ref/export_lpr.py <model.onnx>          # weights + parity ref (once; needs onnx/onnxruntime)
cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m1_lpr.cpp
m1_lpr pure/ref/                                     # -> worst 3.3e-05, argmax 9/9 MATCH
```

## Training (pure C++)
`train_lpr.cpp` trains all 9 heads with summed softmax cross-entropy (Adam). Validated on synthetic
plate samples (random 9-head labels + a label-encoding image): overfitting a small set drives loss
**56.7 → 6.9** and per-head accuracy **5.6% → 80.6%**, confirming forward + 9-head-CE + backward + optimizer.
```sh
cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_lpr.cpp
train_lpr 50 8 3e-4                                  # synthetic: steps batch lr
train_lpr 2000 16 3e-4 --data mydata --save ckpt.bin # REAL folder dataset -> checkpoint
```
**CPU speedup:** add `-DUSE_EIGEN -Ipure/third_party/eigen_flat -arch:AVX2` — conv/matmul route through
the `bk::` seam, **~2× faster** training, same result. **Inference on a real crop** (image → plate
string, pure C++, no Python): `lpr_infer <weights_dir> <plate.jpg>` (built with `/utf-8`) — e.g. a
pipeline crop reads `品川 371 ら 100`.

### Real-data training (`lpr_data.hpp`)
Point `--data <dir>` at a folder with plate crops + a `labels.txt` (one line per image):
```
<filename> <region> <class_num_01> <class_num_02> <class_num_03> <hiragana> <plate_num_01..04>
plate0001.jpg  51 5 0 0 9 1 2 3 4        # indices (0-based) into each head's class list
```
Images are bilinear-resized to 128×128, RGB, /255 (matching `Classificator/lpr/lpr.py`). To make
`labels.txt` from human annotations (`filename ⟶ 品川 ⟶ 500 ⟶ あ ⟶ 1234`, tab-separated), use
`pure/ref/make_labels.py`. Training saves a checkpoint (`--save`); copy `pure/ref/manifest.txt` next
to it and it reloads for inference / the WASM demo. Real JP-plate data is private, so this repo ships
the loader + loop, not the data.

## Roadmap
1. ✅ extract the 2-branch depthwise-separable ResNet arch from the ONNX + forward parity (3.3e-05)
2. ✅ 9-head cross-entropy loss + training loop (synthetic-validated: loss 56.7→6.9, acc 80.6%)
3. ✅ **WebAssembly webcam demo** (recognize → 地域名/分類番号/ひらがな/一連番号)
4. (later) real folder dataset loader + real-data training; Eigen/Thrust backends like the siblings

Reused from the sibling engine: `autograd/backend/ops2d/linalg/bn/optim/ptio/dataset/parallel` +
`face_ops` (relu/gap) + stb + flat Eigen. LPR-specific: `net_lpr.hpp`, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses.
