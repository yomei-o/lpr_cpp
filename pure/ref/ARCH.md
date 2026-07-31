# Japanese LPR classifier — architecture (extracted from the ONNX via export_lpr.py)

Origin: `model_128x128_128_6700_5_4_1_relu.onnx` (Keras/TF → ONNX) from the govtech traffic-count
pipeline `Classificator/lpr`. A depthwise-separable ResNet, all **128 channels**, BN **eps 1e-3**,
conv order **Conv → ReLU → BN** (ReLU *before* BN). Input `[N,128,128,3]` NHWC (the C++ works in
NCHW `[N,3,128,128]`). 27 Conv + 27 BatchNorm + 9 Dense/Softmax heads, ~1.26 MB weights.

## Structure
```
stem:  Conv 4×4 / stride 4  (3→128, no pad)  → ReLU → BN          # 128×128 → 32×32
        │
        ├── branch A (6 blocks) → GlobalAveragePool → feat_A[128]
        │      heads: region_id(133), class_num_01(10), class_num_02(20), class_num_03(22)
        └── branch B (5 blocks) → GlobalAveragePool → feat_B[128]
               heads: hiragana_id(53), plate_num_01(11), plate_num_02(11), plate_num_03(11), plate_num_04(10)
```
Both branches read the **same** stem output (the split point is the stem's BN).

### Block (per branch)
```
block0 :  x = x + BN(ReLU(dwconv5×5(x)))                          # depthwise, groups=128, pad 2; skip = x
block  :  h = BN(ReLU(conv1×1(x)));  x = h + BN(ReLU(dwconv5×5(h)))   # skip = the 1×1 output h
final  :  x = BN(ReLU(conv1×1(x)))
feat   =  GlobalAveragePool(x)                                    # [N,128]
```
Branch A = block0 + 6 blocks + final (14 convs); Branch B = block0 + 5 blocks + final (12 convs);
+ shared stem (1 conv) = 27 convs.

### Heads
Each head is `softmax(feat @ W[128,C] + b)`. Label lists (from `Classificator/lpr/lpr.py`): region =
133 Japanese place names; hiragana = 53 chars (`あいう…` + `ABCEHKLMTYV`); class_num digits; plate_num
digits (value 10 = blank). The decode joins them into the plate string (地域名・分類番号・ひらがな・一連番号).

## Parity
Fixed NHWC input (`sin`-pattern) → onnxruntime 9 heads = `ref/refout.bin`; the pure-C++ forward
(`m1_lpr.cpp`, eval-mode BN) matches to **worst 3.3e-05**, argmax **9/9**.
