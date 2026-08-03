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

## ⏭ 既知の問題 — 直っていない。暇なときに (as of 2026-08-03)
Ordered by how much they hurt. Nothing here blocks the demo; all of it is quality.

1. **The shipped detector weights are the weak link, not the code.** Everything in the table above
   is a property of `yolox/plate_yolox_tiny.onnx`. The classifier reads what the detector misses,
   so effort spent on detection code is wasted — the weights are what need replacing.
   - Tried and **rejected**: RGB-inverting the frame so a black plate looks white. Measured on the
     real photo — original best 0.015, **inverted 0 candidates**. Colour is not the dominant factor.
   - The tell: the runner-up box on the real photo, (313,203)-(387,247) at 0.011, sits *inside* the
     plate and is 74 px wide against the true 296 px — almost exactly ¼. The model carries a scale
     prior from traffic-camera framing and will not stretch to a close-up.
   - Not yet tried: a **scale pyramid** at inference (run 2–4× upscaled tiles, union the boxes).
     ~30 min of work, but given 0.015 there is no evidence it would catch anything, and it costs
     2–4× the runtime. Measure before building it.
2. **Synthetic plate generator — the actual fix, and the biggest single win available.** Render JP
   plates (白/黄/黒/緑, correct 地域名 / 分類番号 / ひらがな / 4桁 layout) and paste them onto
   backgrounds at random scale, rotation and lighting. That yields **detector boxes and the
   classifier's 9 labels at once, unlimited, already labelled** — and lets close-ups and black
   plates be over-sampled, which is exactly what is failing. Both training loops already exist
   (§4, §5); the generator is the only missing piece. Pure C++ with an embedded font keeps the
   repo Python-free. Est. a day.
3. **地域名 is still only 0.62 agreement even with 6-crop TTA** (see above). TTA papers over it; it
   does not fix it. The head is under-trained on small glyphs. Retraining on §2's data with a
   larger input crop for the top strip, or a separate higher-resolution region head, is the fix.
   Until then: trust `conf[0]`, and treat < 0.6 as "unknown region", not as a reading.
4. **`plate.html` recomputes nothing when the slider moves — except the classifier.** Every
   re-filter re-runs TTA (6 forwards ≈ 0.7 s) on each surviving box. Cache by box, keyed on the
   rounded coordinates.
5. **The manual box has no handles.** It can only be redrawn, not nudged — annoying when the crop
   is 5 px off and the region flips. Corner drag handles would help more than they look like they
   would, given §3.
6. **`wasm/test_lpr_box.js` needs a hand-made `.rgba`.** It should take a PNG directly (the C++ side
   already links stb_image; the JS side does not). Low priority, but it is why the test is not in
   any CI-shaped script.
7. **Detector WASM is 3.3 s/frame single-threaded** — button-press only, no live video.
   - Emscripten `-pthread` needs a shared `WebAssembly.Memory`, i.e. a **SharedArrayBuffer**, which
     since Spectre is only handed to a **cross-origin isolated** document: `Cross-Origin-Opener-
     Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` on the *response*. GitHub
     Pages can set no headers at all, and these two do not work as `<meta http-equiv>`.
   - **But that does not make it impossible.** A service worker can intercept its own scope's
     fetches and add the headers itself (`coi-serviceworker`); the page reloads once, then
     `crossOriginIsolated` is true. It works on Pages. Costs: an extra reload on first visit, a
     single-thread fallback for browsers with no service worker, and `require-corp` applying to
     every subresource — harmless here, since every fetch is same-origin and relative.
   - **The reason not to bother is the payoff, not the headers.** Threads give ~2.5–3× on this
     workload, so 3.3 s → ~1.2 s: still a button press, still nowhere near the ~100 ms live video
     needs. Cheaper things to do first, none of which need any header:
     (a) run the detector **in a plain Worker** — no speedup, but 3.3 s of frozen UI becomes 3.3 s
     of responsive page; best effort-to-comfort ratio here;
     (b) **yolox-nano instead of tiny** — 1.08 vs 6.45 GFLOPs @416, ~6×, and §2's retraining is
     where that choice gets made anyway — the single biggest lever;
     (c) 416 → 320 / 256 input (1.7× / 2.6×), accuracy for speed, preview pass only;
     (d) `-mrelaxed-simd` for FMA, ~10–30% on the GEMM, needs a second build for older browsers;
     (e) **measure the GEMM blocking** — `-msimd128` is on but nobody has checked the tile widths
     against wasm's 16 SIMD registers. Profile before optimising, as always.

## Notes / gotchas
- **Two engines**: `pure/` (LPR, facenet-derived, has dtensor GPU) and `yolox/` (detector, own engine)
  have clashing global symbols → cannot co-link → the end-to-end demo is **two WASM modules**.
- The shipped `yolox/plate_yolox_tiny.onnx` is a **ReLU** yolox-tiny; `train_cli` uses standard
  **SiLU** YOLOX — retrain fully from real data, don't mix the ReLU weights.
- Plate = class 7 of `car,small-truck,truck,bus,motorbike,bicycle,person,plate`.
- Live demos: https://yomei-o.github.io/lpr_cpp/wasm/plate.html (detect+recognize) and `/wasm/`
  (classifier only). nvcc build env + emsdk paths: see the facenet_cpp memory.
