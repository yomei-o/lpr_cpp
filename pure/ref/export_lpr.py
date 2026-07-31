# Extract the Japanese-LPR classifier into a manifest + weight blob (forward order) + a fixed-input
# parity reference. Architecture (from graph tracing): a SHARED stem, then the stem output splits
# into TWO branches, each a depthwise-separable ResNet ending in GlobalAveragePool + Dense/Softmax
# heads:
#   stem: Conv4x4/s4(3->128) -> Relu -> BN
#   branch A (6 blocks) -> region_id, class_num_01/02/03
#   branch B (5 blocks) -> hiragana_id, plate_num_01/02/03/04
#   block0 : x = x + BN(Relu(dw5x5(x)))
#   block  : h = BN(Relu(1x1(x))); x = h + BN(Relu(dw5x5(h)))
#   final  : x = BN(Relu(1x1(x)))
#   python export_lpr.py <model.onnx>
import sys, os, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

MP = sys.argv[1] if len(sys.argv) > 1 else \
  r"C:/prog/govtech-traffic-count-pipeline-main/Classificator/lpr/model/R5_01_LPR_30000_10000/model_128x128_128_6700_5_4_1_relu.onnx"
OUT = os.path.dirname(__file__)
m = onnx.load(MP); g = m.graph
init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
prod = {o: i for i, n in enumerate(g.node) for o in n.output}
def attr(n, k, d=None):
    for x in n.attribute:
        if x.name == k: return x.i if k == 'group' else list(x.ints)
    return d

blob = bytearray(); manifest = []
def put(a): blob.extend(np.asarray(a, np.float32).ravel().tobytes())
def emit_conv(n):
    w = init[n.input[1]]; Co, Ci, kh, kw = w.shape
    b = init[n.input[2]] if len(n.input) > 2 and n.input[2] in init else None
    s = attr(n, 'strides', [1, 1]); p = attr(n, 'pads', [0, 0, 0, 0]); grp = attr(n, 'group', 1)
    manifest.append(('C', Co, Ci, kh, kw, s[0], s[1], p[0], p[1], grp, 1 if b is not None else 0))
    put(w); put(b if b is not None else [])
def emit_bn(n):
    manifest.append(('N', init[n.input[1]].shape[0]))
    for k in (1, 2, 3, 4): put(init[n.input[k]])          # gamma, beta, mean, var
def emit_body(lo, hi):                                     # Conv/BN nodes with lo<=idx<=hi, in order
    for i in range(lo, hi + 1):
        n = g.node[i]
        if n.op_type == 'Conv': emit_conv(n)
        elif n.op_type == 'BatchNormalization': emit_bn(n)

# heads: softmax output name -> (W, b)
name2node = {o: n for n in g.node for o in n.output}
def emit_head(out_name):
    sm = name2node[out_name]; add = name2node[sm.input[0]]; mm = name2node[add.input[0]]
    W = init[mm.input[1]]; b = init[add.input[1]]
    manifest.append(('H', W.shape[0], W.shape[1], out_name)); put(W); put(b)

emit_body(1, 3)                                            # shared stem (Conv#1, BN#3)
emit_body(4, 52)                                           # branch A body
for h in ['region_id_output', 'class_num_01_output', 'class_num_02_output', 'class_num_03_output']: emit_head(h)
emit_body(67, 108)                                         # branch B body
for h in ['hiragana_id_output', 'plate_num_01_output', 'plate_num_02_output', 'plate_num_03_output', 'plate_num_04_output']: emit_head(h)

with open(os.path.join(OUT, 'manifest.txt'), 'w') as f:
    f.write(f"{len(manifest)}\n")
    for e in manifest: f.write(' '.join(str(x) for x in e) + '\n')
open(os.path.join(OUT, 'weights.bin'), 'wb').write(blob)

# parity ref (all 9 heads, canonical order matching the C++ forward)
HEADS = ['region_id_output', 'class_num_01_output', 'class_num_02_output', 'class_num_03_output',
         'hiragana_id_output', 'plate_num_01_output', 'plate_num_02_output', 'plate_num_03_output', 'plate_num_04_output']
x = (np.sin(np.arange(128 * 128 * 3).reshape(1, 128, 128, 3) * 0.01) * 0.5 + 0.5).astype(np.float32)
outs = ort.InferenceSession(MP, providers=['CPUExecutionProvider']).run(HEADS, {g.input[0].name: x})
x.transpose(0, 3, 1, 2).astype(np.float32).tofile(os.path.join(OUT, 'input.bin'))    # NCHW for the C++
with open(os.path.join(OUT, 'refout.bin'), 'wb') as f:
    for o in outs: f.write(np.asarray(o, np.float32).ravel().tobytes())
print(f"{len(manifest)} layers, weights {len(blob)/1e6:.2f} MB")
print("head dims:", [o.shape[-1] for o in outs], "argmax:", [int(np.argmax(o)) for o in outs])
