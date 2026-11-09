# Convert yolo_lpr_cpp's from-scratch recognizer (models/plate_ocr_scratch.onnx, BSD-3-Clause, trained
# from random initialisation in lpr_analyzer_cpp) into this repo's manifest + weights.bin + parity ref.
#
# The network is the same one lpr_cpp runs (ARCH.md): shared stem, branch A (block0 + 6 blocks + final),
# branch B (block0 + 5 blocks + final), Conv -> ReLU -> BN with eps 1e-3. It differs only in the heads:
#   * region has 138 classes; rows 0..132 are lpr_cpp's 133 names in the same order (append-only, see
#     yolo_lpr_cpp/spec/labels.txt), so the first 133 rows are kept and the 5 names added in 2025 dropped
#   * plate_kind / legible are extra heads lpr_cpp does not have; they are skipped
# The ONNX was exported from PyTorch, so the layers are found by initializer name, and Gemm weights
# ([out,in], transB=1) are transposed into lpr_cpp's [in,out].
#
#   python pure/ref/convert_scratch.py <plate_ocr_scratch.onnx>
import os, sys, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

MP = sys.argv[1] if len(sys.argv) > 1 else "../yolo_lpr_cpp/models/plate_ocr_scratch.onnx"
OUT = os.path.dirname(os.path.abspath(__file__))
NREGION = 133

m = onnx.load(MP); g = m.graph
init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
convs = {n.input[1]: n for n in g.node if n.op_type == "Conv"}
bns = {n.input[1]: n for n in g.node if n.op_type == "BatchNormalization"}
for n in bns.values():
    eps = [a.f for a in n.attribute if a.name == "epsilon"]
    assert eps and abs(eps[0] - 1e-3) < 1e-9, "BN eps must be 1e-3 (lpr_cpp hard-codes it): %s" % eps

blob = bytearray(); manifest = []
def put(a): blob.extend(np.asarray(a, np.float32).ravel().tobytes())
def attr(n, k, d):
    for x in n.attribute:
        if x.name == k: return x.i if k == "group" else list(x.ints)
    return d
def cbr(prefix):                                    # Conv -> (ReLU) -> BN, by module name
    c = convs[prefix + ".conv.weight"]; w = init[c.input[1]]
    b = init[c.input[2]] if len(c.input) > 2 else None
    s = attr(c, "strides", [1, 1]); p = attr(c, "pads", [0, 0, 0, 0]); grp = attr(c, "group", 1)
    Co, Ci, kh, kw = w.shape
    manifest.append(("C", Co, Ci, kh, kw, s[0], s[1], p[0], p[1], grp, 1 if b is not None else 0))
    put(w); put(b if b is not None else [])
    n = bns[prefix + ".bn.weight"]
    manifest.append(("N", init[n.input[1]].shape[0]))
    for k in (1, 2, 3, 4): put(init[n.input[k]])      # gamma, beta, running mean, running var
def branch(br, nblocks):
    cbr(f"model.{br}.block0")
    for i in range(nblocks): cbr(f"model.{br}.mix.{i}"); cbr(f"model.{br}.dw.{i}")
    cbr(f"model.{br}.final")
def head(src, name, rows=None):
    W = init[f"model.heads.{src}.weight"]; b = init[f"model.heads.{src}.bias"]   # W [out,in]
    if rows is not None: W, b = W[:rows], b[:rows]
    manifest.append(("H", W.shape[1], W.shape[0], name)); put(W.T); put(b)

cbr("model.stem")
branch("a", 6)
head("region", "region_id_output", NREGION)
for h in ("class_num_01", "class_num_02", "class_num_03"): head(h, h + "_output")
branch("b", 5)
head("hiragana", "hiragana_id_output")
for h in ("plate_num_01", "plate_num_02", "plate_num_03", "plate_num_04"): head(h, h + "_output")

with open(os.path.join(OUT, "manifest.txt"), "w", newline="\n") as f:
    f.write(f"{len(manifest)}\n")
    for e in manifest: f.write(" ".join(str(x) for x in e) + "\n")
open(os.path.join(OUT, "weights.bin"), "wb").write(blob)

# Parity ref: the same ONNX with the region head cut to 133 rows (so its softmax runs over the same
# classes as the C++), on the same fixed sinusoid input export_lpr.py uses.
for t in g.initializer:
    if t.name in ("model.heads.region.weight", "model.heads.region.bias"):
        t.CopyFrom(numpy_helper.from_array(init[t.name][:NREGION], t.name))
for o in g.output:
    if o.name == "region": o.type.tensor_type.shape.dim[1].dim_value = NREGION
HEADS = ["region", "class_num_01", "class_num_02", "class_num_03",
         "hiragana", "plate_num_01", "plate_num_02", "plate_num_03", "plate_num_04"]
x = (np.sin(np.arange(128 * 128 * 3).reshape(1, 128, 128, 3) * 0.01) * 0.5 + 0.5).astype(np.float32)
x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))                       # NCHW, as the C++ reads it
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
outs = sess.run(HEADS, {g.input[0].name: x})
x.tofile(os.path.join(OUT, "input.bin"))
with open(os.path.join(OUT, "refout.bin"), "wb") as f:
    for o in outs: f.write(np.asarray(o, np.float32).ravel().tobytes())
print(f"{len(manifest)} layers, weights {len(blob)/1e6:.2f} MB")
print("head dims:", [o.shape[-1] for o in outs], "argmax:", [int(np.argmax(o)) for o in outs])
