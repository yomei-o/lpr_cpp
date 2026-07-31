// LPR training: 9-head softmax cross-entropy + Adam. Two data sources:
//   --data <dir>  : REAL plate crops (dir/labels.txt + images; see lpr_data.hpp) — production path.
//   (default)     : SYNTHETIC samples (random 9 labels + a label-encoding image) — validates the loop
//                   where real JP-plate data isn't available.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_lpr.cpp
//   run:   train_lpr [steps] [batch] [lr] [--data DIR] [--ref pure/ref/] [--save ckpt.bin] [--syn N]
#define STB_IMAGE_IMPLEMENTATION
#include "lpr_data.hpp"
#include "net_lpr.hpp"
#include "lpr_loss.hpp"
#include "optim.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

static std::string opt(int c, char** v, const std::string& k, const std::string& d) {
  for (int i = 1; i < c - 1; ++i) if (k == v[i]) return v[i + 1]; return d;
}
static float head_acc(const LprOut& out, const std::vector<std::vector<int>>& yb, int B) {
  int ok = 0, tot = 0;
  for (int h = 0; h < 9; ++h) { int C = LPR_DIMS[h]; auto& d = out.heads[h]->data;
    for (int b = 0; b < B; ++b) { int a = 0; float m = -1e9f; for (int i = 0; i < C; ++i) if (d[b * C + i] > m) { m = d[b * C + i]; a = i; }
      ok += (a == yb[h][b]); ++tot; } }
  return 100.f * ok / tot;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int steps = argc > 1 && argv[1][0] != '-' ? atoi(argv[1]) : 60;
  int B = argc > 2 && argv[2][0] != '-' ? atoi(argv[2]) : 8;
  float lr = argc > 3 && argv[3][0] != '-' ? (float)atof(argv[3]) : 3e-4f;
  std::string DATA = opt(argc, argv, "--data", ""), RF = opt(argc, argv, "--ref", "pure/ref/");
  std::string SAVE = opt(argc, argv, "--save", ""); int SYN = atoi(opt(argc, argv, "--syn", "32").c_str());
  if (RF.back() != '/') RF += '/';

  LProv p = load_lpr(RF);                                   // pretrained init (trainable)
  std::vector<Tensor> params = lpr_params(p);
  Adam optm(params, lr);
  std::mt19937 rng(1234);

  // real dataset OR synthetic set
  LprDS ds; bool real = !DATA.empty();
  std::vector<std::vector<int>> slab; std::vector<std::vector<float>> simg;
  if (real) { ds = read_lpr_dataset(DATA);
    if (ds.paths.empty()) { printf("no samples in %s (need labels.txt + images)\n", DATA.c_str()); return 1; }
    printf("LPR train: REAL data %zu images, batch %d, lr %.0e\n", ds.paths.size(), B, lr);
  } else {
    slab.assign(9, std::vector<int>(SYN)); simg.assign(SYN, std::vector<float>(3 * 128 * 128));
    std::normal_distribution<float> noise(0, 0.03f);
    for (int s = 0; s < SYN; ++s) { for (int h = 0; h < 9; ++h) slab[h][s] = rng() % LPR_DIMS[h];
      for (int cell = 0; cell < 9; ++cell) { int cr = cell / 3, cc = cell % 3; float v = (slab[cell][s] + 0.5f) / LPR_DIMS[cell];
        for (int y = cr * 42; y < cr * 42 + 42 && y < 128; ++y) for (int x = cc * 42; x < cc * 42 + 42 && x < 128; ++x)
          for (int c = 0; c < 3; ++c) simg[s][(c * 128 + y) * 128 + x] = std::min(1.f, std::max(0.f, v + noise(rng))); } }
    printf("LPR train: SYNTHETIC %d samples, batch %d, lr %.0e\n", SYN, B, lr);
  }
  printf("  %zu param tensors%s\n", params.size(), real ? "" : " (synthetic validates the loop)");

  for (int it = 1; it <= steps; ++it) {
    Tensor x; std::vector<std::vector<int>> yb;
    if (real) { LBatch bt = lpr_sample_batch(ds, B, rng); x = bt.x; yb = bt.lab; }
    else { std::vector<float> xb((size_t)B * 3 * 128 * 128); yb.assign(9, std::vector<int>(B));
      for (int b = 0; b < B; ++b) { int idx = rng() % SYN; std::copy(simg[idx].begin(), simg[idx].end(), xb.begin() + (size_t)b * 3 * 128 * 128);
        for (int h = 0; h < 9; ++h) yb[h][b] = slab[h][idx]; }
      x = from_data({B, 3, 128, 128}, xb); }

    optm.zero_grad();
    LprOut out = lpr_forward(x, p, true, false);            // train mode, logits
    Tensor L = lpr_loss(out, yb);
    backward(L); optm.step();
    if (it % 10 == 0 || it == 1) printf("  step %3d/%d  loss %.4f  acc %.1f%%\n", it, steps, L->data[0], head_acc(out, yb, B));
    free_graph(L);
  }
  if (!SAVE.empty()) { save_lpr(p, SAVE); printf("saved checkpoint -> %s  (copy %smanifest.txt next to it to reload)\n", SAVE.c_str(), RF.c_str()); }
  printf("done.\n");
  return 0;
}
