# lpr_cpp — pure C++ Japanese license-plate classifier (no PyTorch/TF, no CMake) — WIP

The license-plate **recognition** stage of the govtech traffic-count pipeline
(`Classificator/lpr`), ported to a dependency-free C++ autograd engine — **training and inference**,
no Python at run time. Plate **detection** is YOLOX (already ported in
[yolox_cpp](https://github.com/yomei-o/yolox_cpp)); this repo is the per-plate **classifier** that
reads the 地域名・分類番号・ひらがな・一連番号.

Reference model = the pipeline's `model_128x128_..._relu.onnx` (Keras/TF): a depthwise-separable
ResNet, 128 ch, that takes a 128×128 plate crop and emits **9 softmax heads** via two branches.
Full spec: [pure/ref/ARCH.md](pure/ref/ARCH.md).

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

## Roadmap
1. ✅ extract the 2-branch depthwise-separable ResNet arch from the ONNX + forward parity (3.3e-05)
2. **next:** 9-head cross-entropy loss + training loop (synthetic-validated; folder dataset loader)
3. inference CLI + decode to the plate string; **WebAssembly webcam demo** (recognize → show text)
4. (later) real-data training; Eigen/Thrust backends like the sibling repos

Reused from the sibling engine: `autograd/backend/ops2d/linalg/bn/optim/ptio/dataset/parallel` +
`face_ops` (relu/gap) + stb + flat Eigen. LPR-specific: `net_lpr.hpp`, `pure/ref/*`.

License: own code BSD-3-Clause; bundled deps keep their licenses.
