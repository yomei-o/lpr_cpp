// Pure-C++ Japanese license-plate classifier (depthwise-separable ResNet, Keras/TF origin).
// Shared stem -> two branches (A: 6 blocks -> region+class_num×3; B: 5 blocks -> hiragana+plate_num×4)
// -> GlobalAveragePool -> Dense/Softmax heads. Weights loaded in forward order from ref/manifest.txt
// + ref/weights.bin (export_lpr.py). Conv order is Conv -> ReLU -> BN (BN eps 1e-3). See ref/ARCH.md.
#pragma once
#include "autograd.hpp"
#include "face_ops.hpp"     // relu, gap, add_rowvec
#include "bn.hpp"           // batchnorm2d
#include "linalg.hpp"       // matmul
#include "ops2d.hpp"        // softmax_rows
#include <fstream>
#include <string>
#include <vector>

struct LLayer { int kind; int64_t Co, Ci, kh, kw, sh, sw, ph, pw, grp;   // kind 0=conv 1=bn 2=head
                Tensor w, b, gamma, beta; std::vector<float> rm, rv; };
struct LProv { std::vector<LLayer> L; size_t i = 0; float eps = 1e-3f; LLayer& next() { return L[i++]; } };

inline LProv load_lpr(const std::string& dir) {
  std::string D = dir; if (!D.empty() && D.back() != '/') D += '/';
  std::ifstream mf(D + "manifest.txt"); if (!mf) { printf("missing %smanifest.txt\n", D.c_str()); std::exit(1); }
  std::ifstream wf(D + "weights.bin", std::ios::binary); if (!wf) { printf("missing %sweights.bin\n", D.c_str()); std::exit(1); }
  auto rd = [&](int64_t n) { std::vector<float> v(n); wf.read((char*)v.data(), n * 4); return v; };
  int N; mf >> N; LProv p; p.L.reserve(N);
  for (int i = 0; i < N; ++i) {
    std::string k; mf >> k; LLayer L{};
    if (k == "C") { int hb; mf >> L.Co >> L.Ci >> L.kh >> L.kw >> L.sh >> L.sw >> L.ph >> L.pw >> L.grp >> hb; L.kind = 0;
      L.w = from_data({L.Co, L.Ci, L.kh, L.kw}, rd(L.Co * L.Ci * L.kh * L.kw), true);
      if (hb) L.b = from_data({L.Co}, rd(L.Co), true); }
    else if (k == "N") { mf >> L.Co; L.kind = 1; L.gamma = from_data({L.Co}, rd(L.Co), true); L.beta = from_data({L.Co}, rd(L.Co), true);
      L.rm = rd(L.Co); L.rv = rd(L.Co); }
    else { mf >> L.Ci >> L.Co; std::string nm; mf >> nm; L.kind = 2;   // head: W[in,out], b[out]
      L.w = from_data({L.Ci, L.Co}, rd(L.Ci * L.Co), true); L.b = from_data({L.Co}, rd(L.Co), true); }
    p.L.push_back(std::move(L));
  }
  return p;
}

// Conv -> ReLU -> BN
inline Tensor l_cbr(const Tensor& x, LProv& p, bool tr) {
  auto& C = p.next(); Tensor nob;
  Tensor y = conv2d(x, C.w, C.b ? C.b : nob, C.sh, C.ph, C.grp);
  y = relu(y);
  auto& N = p.next();
  return batchnorm2d(y, N.gamma, N.beta, N.rm, N.rv, p.eps, tr, 0.1f);
}
// one branch: block0 (x + dw) -> n blocks (h=1x1; x=h+dw) -> final 1x1 -> GAP -> (N,128)
inline Tensor l_branch(Tensor x, LProv& p, bool tr, int nblocks) {
  x = add(x, l_cbr(x, p, tr));                         // block0: dw with input skip
  for (int i = 0; i < nblocks; ++i) { Tensor h = l_cbr(x, p, tr); x = add(h, l_cbr(h, p, tr)); }
  x = l_cbr(x, p, tr);                                 // final 1x1
  return gap(x);
}
// head: softmax(feat[N,128] @ W[128,C] + b)
inline Tensor l_head(const Tensor& feat, LProv& p) {
  auto& H = p.next(); return softmax_rows(add_rowvec(matmul(feat, H.w), H.b));
}

struct LprOut { std::vector<Tensor> heads; };   // region, class_num_01/02/03, hiragana, plate_num_01..04

inline LprOut lpr_forward(Tensor x, LProv& p, bool tr) {
  p.i = 0;
  Tensor s = l_cbr(x, p, tr);                          // shared stem (Conv4x4/s4 -> ReLU -> BN)
  Tensor fa = l_branch(s, p, tr, 6);                   // branch A
  LprOut o;
  o.heads.push_back(l_head(fa, p));                    // region_id
  o.heads.push_back(l_head(fa, p));                    // class_num_01
  o.heads.push_back(l_head(fa, p));                    // class_num_02
  o.heads.push_back(l_head(fa, p));                    // class_num_03
  Tensor fb = l_branch(s, p, tr, 5);                   // branch B (same stem output)
  o.heads.push_back(l_head(fb, p));                    // hiragana_id
  o.heads.push_back(l_head(fb, p));                    // plate_num_01
  o.heads.push_back(l_head(fb, p));                    // plate_num_02
  o.heads.push_back(l_head(fb, p));                    // plate_num_03
  o.heads.push_back(l_head(fb, p));                    // plate_num_04
  return o;
}
