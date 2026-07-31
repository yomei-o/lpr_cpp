// Pure-C++ parity for the plate detector (ReLU yolox-tiny, 8-class): run plate_yolox_tiny.onnx via
// the graph interpreter, stopping at the 3 per-level head tensors [1,13,H,W], and compare to
// onnxruntime. These heads feed the standard YOLOX decode+NMS (infer_yolox.hpp) — no need to express
// the ONNX decode tail (Reshape/Transpose/3D-Concat) in the 4D engine.
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m_plate_parity.cpp
//   run from lpr_cpp/yolox/ :  m_plate_parity
#include "onnx_run.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
using namespace onx;

static std::vector<float> rd(const std::string& p, int64_t n) {
  std::ifstream f(p, std::ios::binary); std::vector<float> v(n);
  if (!f) { printf("missing %s\n", p.c_str()); std::exit(1); } f.read((char*)v.data(), n * 4); return v;
}
int main(int argc, char** argv) {
  std::string onnx = argc > 1 ? argv[1] : "plate_yolox_tiny.onnx", RF = argc > 2 ? argv[2] : "ref_plate/";
  if (RF.back() != '/') RF += '/';
  Graph g = load_onnx(onnx);
  Tensor x = from_data({1, 3, 416, 416}, rd(RF + "input.bin", 3 * 416 * 416));
  const char* HN[3] = {"/head/Concat_output_0", "/head/Concat_1_output_0", "/head/Concat_2_output_0"};
  int HW[3] = {52, 26, 13};
  std::set<std::string> stop(HN, HN + 3);
  auto vals = run_onnx(g, x, stop);
  float worst = 0;
  for (int i = 0; i < 3; ++i) {
    Tensor y = vals.at(HN[i]); auto ref = rd(RF + "head" + std::to_string(i) + ".bin", 13 * HW[i] * HW[i]);
    float w = 0; for (int64_t j = 0; j < y->numel(); ++j) w = std::max(w, std::fabs(y->data[j] - ref[j]));
    worst = std::max(worst, w);
    printf("  head%d [1,13,%d,%d]  worst %.3e\n", i, HW[i], HW[i], w);
  }
  printf("plate detector (pure-C++ onnx run): worst |pure - onnxruntime| = %.3e  %s\n",
         worst, worst < 1e-3f ? "MATCH" : "MISMATCH");
  return 0;
}
