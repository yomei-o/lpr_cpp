// Plate detector demo (host): image -> YOLOX letterbox(416,pad114,BGR,0-255) -> run the ReLU
// plate yolox-tiny ONNX (pure C++) -> 3 head levels -> decode+NMS -> boxes. Prints detections and
// draws them. class 7 = plate. (End-to-end plate+recognition is a browser 2-WASM demo; see wasm/.)
//   build: cl /std:c++20 /O2 /EHsc /Zc:preprocessor /DNOMINMAX /Ipure\third_party pure\m_plate_detect.cpp
//   run from lpr_cpp/yolox/ :  m_plate_detect assets/bus.jpg out.png [conf]
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "onnx_run.hpp"
#include "infer_yolox.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace onx;

static const char* CLS[8] = {"car", "small-truck", "truck", "bus", "motorbike", "bicycle", "person", "plate"};

int main(int argc, char** argv) {
  std::string src = argc > 1 ? argv[1] : "assets/bus.jpg", outp = argc > 2 ? argv[2] : "plate_det.png";
  float conf = argc > 3 ? (float)atof(argv[3]) : 0.3f; const int S = 416;
  int w0, h0, ch; unsigned char* im = stbi_load(src.c_str(), &w0, &h0, &ch, 3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }

  float scale = std::min((float)S / w0, (float)S / h0); int nw = (int)(w0 * scale), nh = (int)(h0 * scale);
  Tensor x = from_data({1, 3, S, S}, std::vector<float>(3 * S * S, 114.f));   // pad 114
  auto px = [&](int yy, int xx, int c) { yy = std::clamp(yy, 0, h0 - 1); xx = std::clamp(xx, 0, w0 - 1); return (float)im[(yy * w0 + xx) * 3 + c]; };
  for (int y = 0; y < nh; ++y) for (int xx = 0; xx < nw; ++xx) {
    float sy = (y + 0.5f) / scale - 0.5f, sx = (xx + 0.5f) / scale - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) {
      float v = px(y0, x0, c) * (1 - fx) * (1 - fy) + px(y0, x0 + 1, c) * fx * (1 - fy)
              + px(y0 + 1, x0, c) * (1 - fx) * fy + px(y0 + 1, x0 + 1, c) * fx * fy;
      x->data[((2 - c) * S + y) * S + xx] = v;                                 // BGR, 0-255, top-left
    }
  }
  Graph g = load_onnx("plate_yolox_tiny.onnx");
  const char* HN[3] = {"/head/Concat_output_0", "/head/Concat_1_output_0", "/head/Concat_2_output_0"};
  auto vals = run_onnx(g, x, {HN[0], HN[1], HN[2]});
  std::vector<Tensor> raw = {vals.at(HN[0]), vals.at(HN[1]), vals.at(HN[2])};
  // logits=false: this head came from the ONNX, which already sigmoided obj and cls.
  auto dets = yolox_detect(raw, {8, 16, 32}, 8, conf, 0.45f, /*logits=*/false);

  auto put = [&](int a, int b, unsigned char r, unsigned char gg, unsigned char bl) { if (a < 0 || b < 0 || a >= w0 || b >= h0) return; unsigned char* q = &im[(b * w0 + a) * 3]; q[0] = r; q[1] = gg; q[2] = bl; };
  printf("%zu detections (conf>=%.2f):\n", dets.size(), conf);
  for (auto& d : dets) {
    int x1 = (int)(d.x1 / scale), y1 = (int)(d.y1 / scale), x2 = (int)(d.x2 / scale), y2 = (int)(d.y2 / scale);
    x1 = std::clamp(x1, 0, w0 - 1); y1 = std::clamp(y1, 0, h0 - 1); x2 = std::clamp(x2, 0, w0 - 1); y2 = std::clamp(y2, 0, h0 - 1);
    unsigned char cr = d.cls == 7 ? 255 : 60, cg = d.cls == 7 ? 60 : 220;
    for (int t = 0; t < 3; ++t) { for (int a = x1; a <= x2; ++a) { put(a, y1 + t, cr, cg, 60); put(a, y2 - t, cr, cg, 60); } for (int b = y1; b <= y2; ++b) { put(x1 + t, b, cr, cg, 60); put(x2 - t, b, cr, cg, 60); } }
    printf("  %-12s conf=%.2f  xyxy=(%d,%d,%d,%d)\n", CLS[d.cls], d.score, x1, y1, x2, y2);
  }
  stbi_write_png(outp.c_str(), w0, h0, 3, im, w0 * 3); printf("wrote %s\n", outp.c_str());
  stbi_image_free(im); return 0;
}
