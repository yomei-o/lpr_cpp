// LPR classifier inference on a plate crop (pure C++, no Python): load weights (ref/ or a
// trained checkpoint dir) + an image -> 128x128, RGB, /255 -> forward -> 9 argmax -> plate
// string (地域名 分類番号 ひらがな 一連番号).
//
// By default it reads the plate from SEVERAL crops and sums the per-head probabilities
// (pure/lpr_tta.hpp). The region head is 130-odd classes decided by a few small glyphs and it
// flips with a few percent of crop margin, so a single crop is close to a coin flip on it.
//
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /utf-8 /Ipure\third_party pure\lpr_infer.cpp
//     (/utf-8 is required — the label tables in lpr_labels.hpp are UTF-8 Japanese strings)
//   run:   lpr_infer <weights_dir> <plate.png>
//          lpr_infer <weights_dir> <frame.png> --box 252 188 548 348     (crop from a full frame)
//          lpr_infer <weights_dir> <plate.png> --single                  (one crop, old behaviour)
#define STB_IMAGE_IMPLEMENTATION
#include "lpr_data.hpp"        // load_plate
#include "net_lpr.hpp"
#include "lpr_labels.hpp"
#include "lpr_tta.hpp"
#include <cstdio>
#include <cstdlib>
#include <array>
#include <string>

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage: lpr_infer <weights_dir> <image> [--box x0 y0 x1 y1] [--single]\n"
           "  default: multi-crop TTA -- the region head is crop-sensitive, see pure/lpr_tta.hpp\n"
           "  --box:   read the plate from this box of a full frame (default: the whole image)\n"
           "  --single: one crop only, the old behaviour\n");
    return 1;
  }
  const std::string WD = argv[1], img = argv[2];
  bool single = false;
  float bx[4] = {-1.f, -1.f, -1.f, -1.f};
  for (int i = 3; i < argc; ++i) {
    const std::string opt = argv[i];
    if (opt == "--single") single = true;
    else if (opt == "--box" && i + 4 < argc) { for (int k = 0; k < 4; ++k) bx[k] = (float)atof(argv[++i]); }
    else { printf("unknown option %s\n", opt.c_str()); return 1; }
  }

  LProv p = load_lpr(WD);
  int W = 0, H = 0, C = 0;
  unsigned char* im = stbi_load(img.c_str(), &W, &H, &C, 3);
  if (!im) { printf("cannot load %s\n", img.c_str()); return 1; }
  if (bx[0] < 0) { bx[0] = 0.f; bx[1] = 0.f; bx[2] = (float)W; bx[3] = (float)H; }
  printf("%s %dx%d, box (%.0f,%.0f)-(%.0f,%.0f)\n", img.c_str(), W, H, bx[0], bx[1], bx[2], bx[3]);

  std::array<int, 9> a{};
  std::array<float, 9> conf{};
  int crops = 1;

  if (single) {
    std::vector<float> v = lpr::crop_to_input(im, W, H, bx[0], bx[1], bx[2], bx[3], 0.f);
    LprOut out = lpr_forward(from_data({1, 3, 128, 128}, std::move(v)), p, false);
    for (int h = 0; h < 9; ++h) {
      const std::vector<float>& d = out.heads[h]->data;
      int b = 0; float m = -1e30f;
      for (int i = 0; i < (int)d.size(); ++i) if (d[i] > m) { m = d[i]; b = i; }
      a[(size_t)h] = b;
    }
  } else {
    const lpr::TtaOut t = lpr::recognize_tta(im, W, H, bx[0], bx[1], bx[2], bx[3], p);
    a = t.arg; conf = t.conf; crops = t.crops;
  }
  stbi_image_free(im);

  const lpr::Plate pl = lpr::decode(a);
  printf("%s  %s %s %s\n", pl.region.c_str(), pl.cls.c_str(), pl.hira.c_str(),
         pl.num.empty() ? "----" : pl.num.c_str());
  printf("  (%s; argmax region=%d class=%d,%d,%d hira=%d plate=%d,%d,%d,%d)\n",
         single ? "single crop" : ("TTA over " + std::to_string(crops) + " crops").c_str(),
         a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
  if (!single) {
    // The region head's share is the number worth watching: low means the crops disagreed
    // and the reading should not be trusted.
    printf("  agreement: region %.2f  class %.2f/%.2f/%.2f  hira %.2f  num %.2f/%.2f/%.2f/%.2f\n",
           conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6], conf[7], conf[8]);
  }
  return 0;
}
