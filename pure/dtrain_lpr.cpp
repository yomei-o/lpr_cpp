// Device (Thrust) training for the LPR classifier — one source, CPU under THRUST_DEVICE_SYSTEM_CPP /
// GPU under nvcc -DUSE_CUDA. Device fwd (train-mode BN) + bwd + DAdam; the 9-head cross-entropy is
// computed on the host autograd (device head logits -> host leaves -> loss -> inject grads ->
// dbackward_from), mirroring facenet's device-training bridge. Synthetic data validates the loop.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dtrain_lpr.cpp
//   GPU (Colab nvcc): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party pure/dtrain_lpr.cpp -o dtr
#include "dnet_lpr.hpp"        // device engine
#include "autograd.hpp"        // host Tensor
#include "face_loss.hpp"       // host cross_entropy
#include <cstdio>
#include <vector>
#include <random>
#include <thrust/copy.h>

static const int DIMS[9] = {133, 10, 20, 22, 53, 11, 11, 11, 10};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int steps = argc > 1 ? atoi(argv[1]) : 20, B = argc > 2 ? atoi(argv[2]) : 4, S = argc > 3 ? atoi(argv[3]) : 8;
  float lr = argc > 4 ? (float)atof(argv[4]) : 3e-4f;
  std::string RF = argc > 5 ? argv[5] : "pure/ref/"; if (RF.back() != '/') RF += '/';

  DLProv p = dlpr_build(RF);
  std::vector<DT> params = dlpr_params(p);
  DAdam opt(params, lr);

  // synthetic fixed set (same encoding as train_lpr.cpp)
  std::mt19937 rng(0); std::normal_distribution<float> noise(0, 0.03f);
  std::vector<std::vector<int>> slab(9, std::vector<int>(S)); std::vector<std::vector<float>> simg(S, std::vector<float>(3 * 128 * 128));
  for (int s = 0; s < S; ++s) { for (int h = 0; h < 9; ++h) slab[h][s] = rng() % DIMS[h];
    for (int cell = 0; cell < 9; ++cell) { int cr = cell / 3, cc = cell % 3; float v = (slab[cell][s] + 0.5f) / DIMS[cell];
      for (int y = cr * 42; y < cr * 42 + 42 && y < 128; ++y) for (int x = cc * 42; x < cc * 42 + 42 && x < 128; ++x)
        for (int c = 0; c < 3; ++c) simg[s][(c * 128 + y) * 128 + x] = std::min(1.f, std::max(0.f, v + noise(rng))); } }
  printf("device LPR train: %d synthetic samples, batch %d, lr %.0e, %zu param tensors\n", S, B, lr, params.size());

  std::mt19937 pick(1);
  for (int it = 1; it <= steps; ++it) {
    std::vector<float> xb((size_t)B * 3 * 128 * 128); std::vector<std::vector<int>> yb(9, std::vector<int>(B));
    for (int b = 0; b < B; ++b) { int idx = pick() % S; std::copy(simg[idx].begin(), simg[idx].end(), xb.begin() + (size_t)b * 3 * 128 * 128);
      for (int h = 0; h < 9; ++h) yb[h][b] = slab[h][idx]; }

    opt.zero_grad();
    std::vector<DT> heads = dlpr_forward(dfrom({B, 3, 128, 128}, xb), p, true);   // device train-mode logits
    // bridge: device head logits -> host leaves -> 9-head CE
    std::vector<Tensor> hl; Tensor total;
    for (int h = 0; h < 9; ++h) { Tensor t = from_data({B, DIMS[h]}, dto_host(heads[h]), true); hl.push_back(t);
      Tensor ce = cross_entropy(t, yb[h]); total = h == 0 ? ce : add(total, ce); }
    backward(total);
    for (int h = 0; h < 9; ++h) thrust::copy(hl[h]->grad.begin(), hl[h]->grad.end(), heads[h]->grad.begin());   // inject
    dbackward_from(heads); bk::sync();
    opt.step();
    if (it % 5 == 0 || it == 1) printf("  step %3d/%d  loss %.4f\n", it, steps, total->data[0]);
  }
  printf("done.  ");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host, THRUST_DEVICE_SYSTEM_CPP)\n");
#endif
  return 0;
}
