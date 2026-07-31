// Device (Thrust) LPR eval forward parity: argmax of the 9 heads vs the ONNX reference.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dnet_lpr_test.cpp
//   GPU (Colab nvcc): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party pure/dnet_lpr_test.cpp -o dlpr
#include "dnet_lpr.hpp"
#include <cstdio>
#include <fstream>

static const int DIMS[9] = {133, 10, 20, 22, 53, 11, 11, 11, 10};
static const int REF[9]  = {93, 5, 0, 0, 9, 10, 4, 2, 2};   // argmax from export_lpr.py / onnxruntime

int main(int argc, char** argv) {
  std::string RF = argc > 1 ? argv[1] : "pure/ref/"; if (RF.back() != '/') RF += '/';
  DLProv p = dlpr_build(RF);
  std::ifstream fi(RF + "input.bin", std::ios::binary); std::vector<float> xin(3 * 128 * 128);
  if (!fi) { printf("missing %sinput.bin\n", RF.c_str()); return 1; } fi.read((char*)xin.data(), xin.size() * 4);

  std::vector<DT> heads = dlpr_forward(dfrom({1, 3, 128, 128}, xin), p); bk::sync();
  int match = 0;
  for (int h = 0; h < 9; ++h) { auto d = dto_host(heads[h]); int a = 0; float m = -1e9f;
    for (int i = 0; i < DIMS[h]; ++i) if (d[i] > m) { m = d[i]; a = i; }
    match += (a == REF[h]); printf("  head %d argmax device=%d ref=%d %s\n", h, a, REF[h], a == REF[h] ? "" : " <-- MISMATCH"); }
  printf("device LPR eval: argmax %d/9  %s\n", match, match == 9 ? "MATCH" : "MISMATCH");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host, THRUST_DEVICE_SYSTEM_CPP)\n");
#endif
  return 0;
}
