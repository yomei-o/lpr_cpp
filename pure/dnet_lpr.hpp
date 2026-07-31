// Device-resident (Thrust) LPR classifier eval forward — one source, CPU under
// THRUST_DEVICE_SYSTEM_CPP / GPU under nvcc -DUSE_CUDA (+cuBLAS/+cuDNN). Mirrors net_lpr.hpp
// (Conv->ReLU->BN, depthwise groups, 2 branches, 9 heads) using dbn_eval (running stats) so it
// reproduces the ONNX/CPU result. Loads the same ref/manifest.txt + ref/weights.bin.
#pragma once
#include "dface_ops.hpp"    // drelu, dgap, dadd_rowvec, dbn_eval (+ dtensor.hpp: dconv2d groups, dmatmul, dtranspose2d)
#include <fstream>
#include <string>
#include <vector>

struct DLLayer { int kind; int64_t Co, Ci, kh, kw, sh, sw, ph, pw, grp; DT w, b, gamma, beta, rm, rv; };
struct DLProv { std::vector<DLLayer> L; size_t i = 0; float eps = 1e-3f; DLLayer& next() { return L[i++]; } };

inline DLProv dlpr_build(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  std::ifstream mf(D + "manifest.txt"); if (!mf) { printf("missing %smanifest.txt\n", D.c_str()); std::exit(1); }
  std::ifstream wf(D + "weights.bin", std::ios::binary); if (!wf) { printf("missing %sweights.bin\n", D.c_str()); std::exit(1); }
  auto rd = [&](int64_t n) { std::vector<float> v(n); wf.read((char*)v.data(), n * 4); return v; };
  int N; mf >> N; DLProv p; p.L.reserve(N);
  for (int i = 0; i < N; ++i) {
    std::string k; mf >> k; DLLayer L{};
    if (k == "C") { int hb; mf >> L.Co >> L.Ci >> L.kh >> L.kw >> L.sh >> L.sw >> L.ph >> L.pw >> L.grp >> hb; L.kind = 0;
      L.w = dfrom({L.Co, L.Ci, L.kh, L.kw}, rd(L.Co * L.Ci * L.kh * L.kw)); if (hb) L.b = dfrom({L.Co}, rd(L.Co)); }
    else if (k == "N") { mf >> L.Co; L.kind = 1; L.gamma = dfrom({L.Co}, rd(L.Co)); L.beta = dfrom({L.Co}, rd(L.Co));
      L.rm = dfrom({L.Co}, rd(L.Co)); L.rv = dfrom({L.Co}, rd(L.Co)); }
    else { mf >> L.Ci >> L.Co; std::string nm; mf >> nm; L.kind = 2;
      L.w = dfrom({L.Ci, L.Co}, rd(L.Ci * L.Co)); L.b = dfrom({L.Co}, rd(L.Co)); }
    p.L.push_back(std::move(L));
  }
  return p;
}

inline DT dl_cbr(DT x, DLProv& p, bool tr) {               // Conv -> ReLU -> BN (train=batch stats / eval=running)
  auto& C = p.next(); DT y = dconv2d(x, C.w, C.b, C.sh, C.ph, C.grp);
  y = drelu(y); auto& N = p.next();
  return tr ? dbn(y, N.gamma, N.beta, p.eps, N.rm, N.rv, 0.1f) : dbn_eval(y, N.gamma, N.beta, N.rm, N.rv, p.eps);
}
inline DT dl_branch(DT x, DLProv& p, int nblocks, bool tr) {
  x = dadd(x, dl_cbr(x, p, tr));                            // block0
  for (int i = 0; i < nblocks; ++i) { DT h = dl_cbr(x, p, tr); x = dadd(h, dl_cbr(h, p, tr)); }
  x = dl_cbr(x, p, tr);                                     // final 1x1
  return dgap(x);                                           // (N,128)
}
inline DT dl_head(DT feat, DLProv& p) { auto& H = p.next(); return dadd_rowvec(dmatmul(feat, H.w), H.b); }  // logits

// full forward -> 9 device logit tensors (region, class_num_01/02/03, hiragana, plate_num_01..04)
inline std::vector<DT> dlpr_forward(DT x, DLProv& p, bool tr = false) {
  p.i = 0;
  DT s = dl_cbr(x, p, tr);
  DT fa = dl_branch(s, p, 6, tr); std::vector<DT> o;
  for (int i = 0; i < 4; ++i) o.push_back(dl_head(fa, p));
  DT fb = dl_branch(s, p, 5, tr);
  for (int i = 0; i < 5; ++i) o.push_back(dl_head(fb, p));
  return o;
}

// trainable device params (conv w/b, BN affine, head w/b) for DAdam.
inline std::vector<DT> dlpr_params(DLProv& p) {
  std::vector<DT> ps;
  for (auto& L : p.L) { if (L.w) ps.push_back(L.w); if (L.b) ps.push_back(L.b);
    if (L.gamma) ps.push_back(L.gamma); if (L.beta) ps.push_back(L.beta); }
  return ps;
}
