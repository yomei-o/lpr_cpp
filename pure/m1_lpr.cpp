// Parity: pure-C++ LPR classifier forward vs the ONNX (onnxruntime) reference, all 9 head outputs.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m1_lpr.cpp
//   run:   m1_lpr [ref_dir]
#include "net_lpr.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>

static const int DIMS[9] = {133, 10, 20, 22, 53, 11, 11, 11, 10};
static const char* NAMES[9] = {"region", "class_num_01", "class_num_02", "class_num_03",
                               "hiragana", "plate_num_01", "plate_num_02", "plate_num_03", "plate_num_04"};

int main(int argc, char** argv) {
  std::string RF = argc > 1 ? argv[1] : "pure/ref/"; if (RF.back() != '/') RF += '/';
  LProv p = load_lpr(RF);
  std::ifstream fi(RF + "input.bin", std::ios::binary); std::vector<float> xin(1 * 3 * 128 * 128);
  if (!fi) { printf("missing %sinput.bin\n", RF.c_str()); return 1; } fi.read((char*)xin.data(), xin.size() * 4);
  std::ifstream fo(RF + "refout.bin", std::ios::binary);

  LprOut out = lpr_forward(from_data({1, 3, 128, 128}, xin), p, false);   // eval mode
  float worst = 0; int argmatch = 0;
  for (int h = 0; h < 9; ++h) {
    std::vector<float> ref(DIMS[h]); fo.read((char*)ref.data(), DIMS[h] * 4);
    auto& e = out.heads[h]->data;
    float w = 0; int am_p = 0, am_r = 0; float mp = -1e9, mr = -1e9;
    for (int i = 0; i < DIMS[h]; ++i) { w = std::max(w, std::fabs(e[i] - ref[i]));
      if (e[i] > mp) { mp = e[i]; am_p = i; } if (ref[i] > mr) { mr = ref[i]; am_r = i; } }
    worst = std::max(worst, w); argmatch += (am_p == am_r);
    printf("  %-13s worst %.2e  argmax pure=%d ref=%d %s\n", NAMES[h], w, am_p, am_r, am_p == am_r ? "" : "  <-- MISMATCH");
  }
  printf("LPR forward: worst |pure - onnxruntime| = %.3e   argmax %d/9   %s\n",
         worst, argmatch, worst < 1e-3f ? "MATCH" : "MISMATCH");
  return 0;
}
