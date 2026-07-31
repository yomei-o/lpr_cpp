// Real-data loader for LPR training: a folder with plate crops + a labels.txt mapping each image to
// its 9 head labels (indices). Images are bilinear-resized to 128×128, RGB, /255, NCHW — matching the
// pipeline's preprocessing (Classificator/lpr/lpr.py: resize, RGB, /255). Requires
// STB_IMAGE_IMPLEMENTATION in the including .cpp.
//
// labels.txt format (one line per sample; '#' comments allowed):
//   <filename> <region> <class_num_01> <class_num_02> <class_num_03> <hiragana> <plate_num_01..04>
//   e.g.   plate0001.jpg  51 5 0 0 9 1 2 3 4        # (indices, not text — see ref/make_labels.py)
#pragma once
#include "autograd.hpp"
#include "stb_image.h"
#include <string>
#include <vector>
#include <array>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

static const int LPR_DIMS[9] = {133, 10, 20, 22, 53, 11, 11, 11, 10};

struct LprDS { std::string root; std::vector<std::string> paths; std::vector<std::array<int, 9>> labels; };

inline LprDS read_lpr_dataset(const std::string& dir) {
  LprDS ds; ds.root = dir; if (!ds.root.empty() && ds.root.back() != '/') ds.root += '/';
  std::ifstream f(ds.root + "labels.txt");
  if (!f) { printf("missing %slabels.txt\n", ds.root.c_str()); return ds; }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line); std::string fn; if (!(ss >> fn)) continue;
    std::array<int, 9> L{}; bool ok = true; for (int i = 0; i < 9; ++i) if (!(ss >> L[i])) { ok = false; break; }
    if (!ok) continue;
    for (int i = 0; i < 9; ++i) L[i] = std::min(std::max(L[i], 0), LPR_DIMS[i] - 1);
    ds.paths.push_back(ds.root + fn); ds.labels.push_back(L);
  }
  return ds;
}

// one image -> [3*128*128] RGB, bilinear-resized, /255, NCHW.
inline std::vector<float> load_plate(const std::string& path) {
  int w, h, c; unsigned char* im = stbi_load(path.c_str(), &w, &h, &c, 3);
  std::vector<float> out(3 * 128 * 128, 0.f);
  if (!im) { printf("warn: cannot load %s (zeros)\n", path.c_str()); return out; }
  for (int y = 0; y < 128; ++y) for (int x = 0; x < 128; ++x) {
    float sy = (y + 0.5f) * h / 128 - 0.5f, sx = (x + 0.5f) * w / 128 - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    auto px = [&](int yy, int xx, int ch) { yy = std::clamp(yy, 0, h - 1); xx = std::clamp(xx, 0, w - 1); return (float)im[(yy * w + xx) * 3 + ch]; };
    for (int ch = 0; ch < 3; ++ch) {
      float v = px(y0, x0, ch) * (1 - fx) * (1 - fy) + px(y0, x0 + 1, ch) * fx * (1 - fy)
              + px(y0 + 1, x0, ch) * (1 - fx) * fy + px(y0 + 1, x0 + 1, ch) * fx * fy;
      out[(ch * 128 + y) * 128 + x] = v / 255.f;
    }
  }
  stbi_image_free(im); return out;
}

struct LBatch { Tensor x; std::vector<std::vector<int>> lab; };   // lab[9][B]

inline LBatch lpr_sample_batch(const LprDS& ds, int B, std::mt19937& rng) {
  std::vector<float> xb((size_t)B * 3 * 128 * 128);
  std::vector<std::vector<int>> lab(9, std::vector<int>(B));
  for (int b = 0; b < B; ++b) {
    int idx = ds.paths.empty() ? 0 : (int)(rng() % ds.paths.size());
    auto v = load_plate(ds.paths[idx]); std::copy(v.begin(), v.end(), xb.begin() + (size_t)b * 3 * 128 * 128);
    for (int h = 0; h < 9; ++h) lab[h][b] = ds.labels[idx][h];
  }
  return {from_data({B, 3, 128, 128}, xb), lab};
}
