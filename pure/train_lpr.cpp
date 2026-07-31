// LPR training-loop validation on SYNTHETIC data (real JP-plate data is private). Each sample gets
// random 9-head labels and a 128x128 image that deterministically encodes them (a 3x3 grid of
// label-brightness cells + noise), so the mapping is learnable. Trains the net (9-head cross-entropy,
// Adam) and reports loss + per-head accuracy — validating forward+loss+backward+optimizer end to end.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\train_lpr.cpp
//   run:   train_lpr [steps] [n_samples] [batch] [lr] [ref_dir]
#include "net_lpr.hpp"
#include "lpr_loss.hpp"
#include "optim.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

static const int DIMS[9] = {133, 10, 20, 22, 53, 11, 11, 11, 10};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int steps = argc > 1 ? atoi(argv[1]) : 60, S = argc > 2 ? atoi(argv[2]) : 32, B = argc > 3 ? atoi(argv[3]) : 8;
  float lr = argc > 4 ? (float)atof(argv[4]) : 1e-3f;
  std::string RF = argc > 5 ? argv[5] : "pure/ref/"; if (RF.back() != '/') RF += '/';

  // fixed synthetic set: labels + images that encode them
  std::mt19937 rng(0); std::normal_distribution<float> noise(0, 0.03f);
  std::vector<std::vector<int>> lab(9, std::vector<int>(S));
  std::vector<std::vector<float>> img(S, std::vector<float>(3 * 128 * 128));
  for (int s = 0; s < S; ++s) {
    for (int h = 0; h < 9; ++h) lab[h][s] = rng() % DIMS[h];
    for (int cell = 0; cell < 9; ++cell) {                       // 3x3 grid, each cell = one head's brightness
      int cr = cell / 3, cc = cell % 3; float v = (lab[cell][s] + 0.5f) / DIMS[cell];
      for (int y = cr * 42; y < cr * 42 + 42 && y < 128; ++y) for (int x = cc * 42; x < cc * 42 + 42 && x < 128; ++x)
        for (int c = 0; c < 3; ++c) img[s][(c * 128 + y) * 128 + x] = std::min(1.f, std::max(0.f, v + noise(rng)));
    }
  }

  LProv p = load_lpr(RF);                                        // pretrained init (trainable)
  std::vector<Tensor> params = lpr_params(p);
  Adam opt(params, lr);
  printf("LPR synthetic train: %d samples, batch %d, lr %.0e, %d params tensors\n", S, B, lr, params.size());

  std::mt19937 pick(1);
  for (int it = 1; it <= steps; ++it) {
    std::vector<int> idx(B); for (int b = 0; b < B; ++b) idx[b] = pick() % S;
    std::vector<float> xb(B * 3 * 128 * 128);
    std::vector<std::vector<int>> yb(9, std::vector<int>(B));
    for (int b = 0; b < B; ++b) { std::copy(img[idx[b]].begin(), img[idx[b]].end(), xb.begin() + (size_t)b * 3 * 128 * 128);
      for (int h = 0; h < 9; ++h) yb[h][b] = lab[h][idx[b]]; }

    opt.zero_grad();
    LprOut out = lpr_forward(from_data({B, 3, 128, 128}, xb), p, true, false);   // train mode, logits
    Tensor L = lpr_loss(out, yb);
    backward(L); opt.step();

    if (it % 10 == 0 || it == 1) {
      int correct = 0, tot = 0;                                  // per-head argmax accuracy on this batch
      for (int h = 0; h < 9; ++h) { int C = DIMS[h]; auto& d = out.heads[h]->data;
        for (int b = 0; b < B; ++b) { int a = 0; float m = -1e9f; for (int i = 0; i < C; ++i) if (d[b * C + i] > m) { m = d[b * C + i]; a = i; }
          correct += (a == yb[h][b]); ++tot; } }
      printf("  step %3d/%d  loss %.4f  acc %.1f%%\n", it, steps, L->data[0], 100.0 * correct / tot);
    }
    free_graph(L);
  }
  printf("done (synthetic — validates the 9-head training loop; plug a real folder dataset loader for production)\n");
  return 0;
}
