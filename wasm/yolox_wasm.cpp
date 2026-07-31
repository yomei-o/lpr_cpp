// Plate DETECTOR compiled to WASM (ReLU yolox-tiny, 8-class). fn_detect takes a camera RGBA frame,
// YOLOX-letterboxes it to 416, runs the plate ONNX (pure-C++ graph interpreter), decodes+NMS, and
// returns the plate boxes (class 7) in the frame's own pixel coords. Chained in the browser with the
// LPR classifier (lpr.wasm). The ONNX is fetched into MEMFS at /plate/model.onnx by the page.
#include "onnx_run.hpp"
#include "infer_yolox.hpp"
#include <emscripten/emscripten.h>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace onx;

static Graph* g_graph = nullptr;
static float g_boxes[64 * 5];                              // up to 64 plates: x1,y1,x2,y2,score (frame coords)

extern "C" {
EMSCRIPTEN_KEEPALIVE int fn_ready() { if (!g_graph) g_graph = new Graph(load_onnx("/plate/model.onnx")); return g_graph ? 1 : 0; }

// rgba: w*h*4. Returns number of plate boxes; fills g_boxes. conf/nms are ×100 ints for a simple ABI.
EMSCRIPTEN_KEEPALIVE int fn_detect(unsigned char* rgba, int w, int h, int conf100) {
  if (!g_graph) fn_ready();
  const int S = 416; float scale = std::min((float)S / w, (float)S / h);
  int nw = (int)(w * scale), nh = (int)(h * scale);
  Tensor x = from_data({1, 3, S, S}, std::vector<float>(3 * S * S, 114.f));
  auto px = [&](int yy, int xx, int c) { yy = std::clamp(yy, 0, h - 1); xx = std::clamp(xx, 0, w - 1); return (float)rgba[(yy * (size_t)w + xx) * 4 + c]; };
  for (int y = 0; y < nh; ++y) for (int xx = 0; xx < nw; ++xx) {
    float sy = (y + 0.5f) / scale - 0.5f, sx = (xx + 0.5f) / scale - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx); float fy = sy - y0, fx = sx - x0;
    for (int c = 0; c < 3; ++c) {
      float v = px(y0, x0, c) * (1 - fx) * (1 - fy) + px(y0, x0 + 1, c) * fx * (1 - fy)
              + px(y0 + 1, x0, c) * (1 - fx) * fy + px(y0 + 1, x0 + 1, c) * fx * fy;
      x->data[((2 - c) * S + y) * S + xx] = v;              // BGR, 0-255, top-left letterbox
    }
  }
  const char* HN[3] = {"/head/Concat_output_0", "/head/Concat_1_output_0", "/head/Concat_2_output_0"};
  auto vals = run_onnx(*g_graph, x, {HN[0], HN[1], HN[2]});
  std::vector<Tensor> raw = {vals.at(HN[0]), vals.at(HN[1]), vals.at(HN[2])};
  auto dets = yolox_detect(raw, {8, 16, 32}, 8, conf100 / 100.f, 0.45f);
  int n = 0;
  for (auto& d : dets) { if (d.cls != 7 || n >= 64) continue;                  // class 7 = plate
    g_boxes[n * 5 + 0] = d.x1 / scale; g_boxes[n * 5 + 1] = d.y1 / scale;
    g_boxes[n * 5 + 2] = d.x2 / scale; g_boxes[n * 5 + 3] = d.y2 / scale; g_boxes[n * 5 + 4] = d.score; ++n; }
  return n;
}
EMSCRIPTEN_KEEPALIVE float* fn_boxes() { return g_boxes; }
}
