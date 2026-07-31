# RESUME — remaining work

Status of the pure-C++ YOLOX training toolchain and what's left to make it a full
replacement for reference-quality training. Verified items live in [README.md](README.md);
this file is the forward-looking TODO.

## Done (pure C++, no Python at run time)
- Engine + YOLOX (Focus stem, SimOTA, decoupled anchor-free head), all sizes
  (nano/tiny/s/m/l/x): forward / loss / train / infer / mAP / `.pt`.
- Real training CLI (`pure/train_cli.cpp`): dataset scan → shuffled mini-batches → epochs →
  SimOTA → loss → Adam(warmup+cosine+wd) → per-epoch val mAP@0.5 → `best.pt`/`last.pt`.
- Initial-weight `.pt` generated in C++ (`pure/make_init_pt.cpp`, `rand`/`from`), all sizes,
  zero-Python bootstrap; `best.pt` loads back into the reference YOLOX (0 unexpected).
- **Standard-YOLO dataset ingestion** — directory scan (`images/`↔`labels/`), normalised
  `cls xc yc w h` labels, arbitrary-size images letterboxed (`pure/dataset.hpp`
  `read_yolo_dataset` / `load_boxes_orig`).
- **Augmentation** — mosaic + mixup + random-affine (rotate/scale/shear/translate) + HSV +
  flip, with **close-mosaic** (disable for last N epochs). `AugCfg` / CLI flags.
- **Unified `yolo` CLI** (`pure/yolo.cpp`) reading `data.yaml`: `train` / `val` / `detect`
  (`export` delegates to the standalone ONNX exporter — see remaining #3). Val reports
  **mAP@0.5 and mAP@0.5:0.95**.
- GPU/CUDA seam (`pure/backend.hpp`); conv/matmul route through `bk::`.

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against the reference YOLOX. Only synthetic data checked so far.
2. **Custom `nc`** — head is fixed at 80 classes; `nc != 80` needs the cls head resized +
   re-initialised. Today class ids must be < 80.
3. **`export` in the unified CLI** — fold BN from the `.pt` and emit ONNX in-CLI (today
   `yolo export` points at the standalone, onnxruntime-verified exporter).
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale, label
   smoothing, warmup-bias-lr. (mAP@0.5:0.95 in val — done.)
5. **Speed** — YOLOX forward is **per-image, summed per mini-batch** (SimOTA/loss are
   per-image); batch it like yolov8 for a real speedup.
6. **CPU speed on Apple Silicon** — CUDA seam doesn't help on Mac (Metal≠CUDA); add a BLAS
   path (Apple Accelerate / OpenBLAS) to `bk::gemm_hosted` for a big no-GPU CPU speedup.

## Notes / gotchas
- Coords: everything is xyxy in the **letterboxed SxS pixel** space; GT and detections share
  it. `load_boxes_orig` reads either label format into original pixels, then `lb_map` applies
  the letterbox transform.
- YOLOX eval uses a low conf threshold (~0.05) since `score = obj·cls` runs lower than v8.
- `.pth` init: `yolox_tiny.pth` is a Megvii `{'model': state_dict}` checkpoint — read in C++
  via `pt::load_pt_state_under("model")`. Non-ASCII filesystem paths mangle native argv, so
  keep the `.pth` at an ASCII path (or use `rand` init, which needs no path).
- Build: MSVC via `C:/prog/claude/cc11.sh`; `scratch/` must pre-exist.

7. **Verify the unified `train_cli`/`yolo.cpp` under a CUDA build** — compile with `nvcc -DUSE_CUDA` and run COCO128 end-to-end on a (free-Colab) T4. The CUDA seam + a training loop were verified on T4, but the new dataset-ingestion + augmentation CLI path has not been built/run under nvcc yet (aug/dataset are host-side; conv/matmul auto-route to `bk::` on GPU). Est. COCO128/640px/100ep: T4 GPU ~7-20 min; CPU ~a day (measured ~5.7 s/image fwd+bwd at 640px, naive GEMM) so a real GPU is the fix.

## Eigen + cuDNN compute backends (ported from yolov8_cpp)  ✅ DONE 2026-07-25

Ported and (CPU side) verified locally: Eigen gemm 6.1×, grouped/depthwise grad-check OK,
full-net dnetx_test MATCH (1.6e-3), nvcc -DUSE_CUDA clean, train_cli -DUSE_EIGEN builds. cuDNN
uses `cudnnSetConvolutionGroupCount` for the grouped/depthwise convs; verify on Colab via
`colab/dnet_cudnn_test.ipynb`. Original porting notes below.

yolov8_cpp gained two **opt-in** compute backends (both on main, verified on a Colab T4). The
seam is identical across all repos (`pure/backend.hpp` `bk::gemm*` + `pure/dtensor.hpp` `dconv2d`),
so port both here. Reference: yolov8_cpp commits "optional Eigen GEMM backend" and "optional cuDNN
conv backend", plus `colab/dnet_cudnn_test.ipynb`.

### A. Eigen GEMM (CPU, `-DUSE_EIGEN`) — fast CPU training, zero external dependency
1. Copy the vendored flat Eigen folder from yolov8_cpp: `pure/third_party/eigen_flat/`
   (130 headers, no subdirs, entry header `Eigen_Core.h`, MPL2). It is fully self-contained.
2. In `pure/backend.hpp`: (a) under the CPU `#else` include block add
   `#ifdef USE_EIGEN  #include "Eigen_Core.h"  #endif`; (b) in the CPU section add an
   `#elif defined(USE_EIGEN)` variant of `gemm` / `gemm_nt` / `gemm_tn` that maps the row-major
   float pointers to `Eigen::Map` and does `Cm.noalias() = Am * Bm` (nt/tn use `.transpose()`;
   beta!=0 → `Cm *= beta; Cm.noalias() += …`). Copy verbatim from yolov8_cpp — it is engine-agnostic.
3. Build train_cli with `-DUSE_EIGEN -Ipure/third_party/eigen_flat` + `/arch:AVX2` (cl) or
   `-march=native` (g++). The scratch engine's conv2d/matmul route through
   `bk::gemm_hosted → bk::gemm`, so train_cli picks it up automatically.
   Verify: grad-check clean; `pure/bench_gemm.cpp` (copy from v8) shows ~10-12× over baseline.
   v8 measured: gemm 12.5×, train_cli end-to-end ~3×, parity 1e-6.

### B. cuDNN conv (GPU, `-DUSE_CUDNN`) — fastest GPU training
1. In `pure/dtensor.hpp`: add the cuDNN header/handle block (`bk::bk_cudnn()` + `CUDNN_CHECK`)
   guarded by `#if defined(USE_CUDA) && defined(USE_CUDNN)`, and an **early-return** cuDNN path at
   the top of `dconv2d`: fwd `cudnnConvolutionForward` (+ `cudnnAddTensor` for bias); backward
   `cudnnConvolutionBackwardData` (dIn) / `BackwardFilter` (dW) / `BackwardBias` (dBias). Use
   `CUDNN_CROSS_CORRELATION` (matches im2col, no kernel flip) and **beta=1** on the bwd calls so
   grads accumulate like the existing `+=`. Copy from yolov8_cpp dtensor.hpp.
2. **[yolox] `dconv2d` takes `groups`** → after `cudnnSetConvolution2dDescriptor` add
   `cudnnSetConvolutionGroupCount(cvd, (int)groups)`. Filter Cin = `w->shape[1]` (already per-group);
   depthwise = `groups==Cout`. Focus (`dfocus`) is a pixel-shuffle reshape, NOT a conv — leave it.
3. No cudnn.h on the dev machine → build/verify on Colab using v8's `colab/dnet_cudnn_test.ipynb`
   as a template (it auto-detects the cuDNN header/lib from Colab's torch-bundled cuDNN).
   Expect: `dnet*_test` forward MATCH; training loss decreases like the cuBLAS build.
   v8 measured (T4, COCO128 640 b4): cuBLAS 88.7 → **cuDNN 36.0 s/epoch (2.46×)**.

### Guardrails
- Both flags are opt-in add-ons. The default builds (scratch CPU / Thrust-CPU / plain
  `-DUSE_CUDA`) must stay behaviorally identical — verify they still compile after the edits
  (`run_all.sh`; and a `nvcc -DUSE_CUDA` compile WITHOUT `-DUSE_CUDNN` to confirm the guard is clean).
- Add the "Compute backends — one source, five builds" table to README (copy from yolov8_cpp).
