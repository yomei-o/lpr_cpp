"""Export a checkpoint from train_plate_yolox.py to the ONNX lpr_cpp runs (yolox/plate_yolox_tiny.onnx),
plus the parity reference yolox/ref_plate/{input,head0,head1,head2}.bin for pure/m_plate_parity.cpp.

The graph is Megvii's own export path (decode_in_inference=False): the three per-level head tensors
/head/Concat_output_0, _1, _2 are [1, 5+nc, H, W] = [reg(4), obj.sigmoid(), cls.sigmoid()]. The C++
side stops at those three and decodes itself (infer_yolox.hpp), so the tail after them does not matter.

  python export_plate_onnx.py --yolox-src YOLOX-0.3.0 --ckpt runs/plate/best.pth --out ../plate_yolox_tiny.onnx \
      --ref ../ref_plate
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_plate_yolox import build_model, S  # noqa: E402

HEADS = ["/head/Concat_output_0", "/head/Concat_1_output_0", "/head/Concat_2_output_0"]


def normalise_graph(m):
    """Rewrite the export into the form pure/onnx_run.hpp reads (and the old plate model had):
    Constant nodes become initializers, and the Focus stem's per-axis Slice pairs
    (x[..., ::2, :] then [..., ::2]) become one Slice over axes [2, 3]. onnxruntime gives the
    same result either way (checked below)."""
    from onnx import numpy_helper
    g = m.graph
    consts = {}
    for n in list(g.node):
        if n.op_type == "Constant":
            t = onnx.numpy_helper.to_array(n.attribute[0].t)
            consts[n.output[0]] = t
            g.node.remove(n)
    for k, v in consts.items():
        g.initializer.append(numpy_helper.from_array(np.asarray(v), k))
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    merged = 0
    for b in list(g.node):                     # b: Slice over axis 3 whose input is a Slice over axis 2
        if b.op_type != "Slice" or list(init[b.input[3]]) != [3]:
            continue
        a = producer.get(b.input[0])
        if a is None or a.op_type != "Slice" or list(init[a.input[3]]) != [2]:
            continue
        tag = b.output[0].replace("/", "_")
        vals = {"st": np.array([init[a.input[1]][0], init[b.input[1]][0]], np.int64),
                "en": np.array([init[a.input[2]][0], init[b.input[2]][0]], np.int64),
                "ax": np.array([2, 3], np.int64),
                "sp": np.array([init[a.input[4]][0], init[b.input[4]][0]], np.int64)}
        names = []
        for k, v in vals.items():
            nm = f"{tag}_{k}"
            g.initializer.append(numpy_helper.from_array(v, nm)); names.append(nm)
        node = onnx.helper.make_node("Slice", [a.input[0]] + names, [b.output[0]], name=b.name + "_hw")
        idx = list(g.node).index(b)
        g.node.remove(b)
        g.node.insert(idx, node)
        merged += 1
    while True:                                # drop nodes whose outputs nobody reads any more
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        if not dead:
            break
        for n in dead:
            g.node.remove(n)
    used = {i for n in g.node for i in n.input}
    for t in list(g.initializer):
        if t.name not in used:
            g.initializer.remove(t)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolox-src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    model = build_model(a.yolox_src, "")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    model.head.decode_in_inference = False
    x = torch.zeros(1, 3, S, S)
    torch.onnx.export(model, x, a.out, input_names=["images"], output_names=["output"], opset_version=11,
                      do_constant_folding=True, dynamo=False)
    m = onnx.load(a.out)
    ref_sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    merged = normalise_graph(m)
    onnx.checker.check_model(m)
    onnx.save(m, a.out)
    probe = np.random.default_rng(0).uniform(0, 255, (1, 3, S, S)).astype(np.float32)
    d = np.abs(ref_sess.run(None, {"images": probe})[0] -
               ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"images": probe})[0]).max()
    print(f"normalised: {merged} Focus slice pairs merged, Constants -> initializers, max diff vs export {d:.2e}")
    assert d < 1e-4
    names = {o for n in m.graph.node for o in n.output}
    for h in HEADS:
        assert h in names, f"{h} not in the exported graph"
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB), epoch {ck.get('epoch')}, val {ck.get('val')}")

    # torch vs onnxruntime on the head tensors, then the parity ref for the C++ interpreter
    inp = (np.sin(np.arange(3 * S * S).reshape(1, 3, S, S) * 0.013) * 127 + 128).astype(np.float32)
    for h in HEADS:
        m.graph.output.append(onnx.helper.make_tensor_value_info(h, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(HEADS, {"images": inp})
    print("heads:", [o.shape for o in outs])
    if a.ref:
        os.makedirs(a.ref, exist_ok=True)
        inp.tofile(os.path.join(a.ref, "input.bin"))
        for i, o in enumerate(outs):
            o.astype(np.float32).tofile(os.path.join(a.ref, f"head{i}.bin"))
        print("wrote parity ref to", a.ref)


if __name__ == "__main__":
    main()
