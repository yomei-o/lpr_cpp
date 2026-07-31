// LPR classifier inference on a real plate crop (pure C++, no Python): load weights (ref/ or a
// trained checkpoint dir) + an image -> resize 128, RGB, /255 -> forward -> 9 argmax -> plate string
// (地域名 分類番号 ひらがな 一連番号).
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /utf-8 /Ipure\third_party pure\lpr_infer.cpp
//     (/utf-8 is required — the label tables in lpr_labels.hpp are UTF-8 Japanese strings)
//   run:   lpr_infer <weights_dir> <plate.jpg>          (weights_dir has manifest.txt + weights.bin)
#define STB_IMAGE_IMPLEMENTATION
#include "lpr_data.hpp"        // load_plate
#include "net_lpr.hpp"
#include "lpr_labels.hpp"
#include <cstdio>
#include <array>

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: lpr_infer <weights_dir> <plate.jpg>\n"); return 1; }
  std::string WD = argv[1], img = argv[2];
  LProv p = load_lpr(WD);
  std::vector<float> v = load_plate(img);                 // resize 128, RGB, /255, NCHW
  LprOut out = lpr_forward(from_data({1, 3, 128, 128}, v), p, false);
  std::array<int, 9> a{};
  for (int h = 0; h < 9; ++h) { auto& d = out.heads[h]->data; int b = 0; float m = -1e9f;
    for (int i = 0; i < (int)d.size(); ++i) if (d[i] > m) { m = d[i]; b = i; } a[h] = b; }
  lpr::Plate pl = lpr::decode(a);
  printf("%s  %s %s %s\n", pl.region.c_str(), pl.cls.c_str(), pl.hira.c_str(), pl.num.empty() ? "----" : pl.num.c_str());
  printf("  (argmax: region=%d class=%d,%d,%d hira=%d plate=%d,%d,%d,%d)\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
  return 0;
}
