// Japanese LPR classifier compiled to WebAssembly. The page crops the on-screen plate guide box to a
// 128×128 RGBA canvas and calls fn_recognize; this standardizes (/255, RGB, NCHW), runs the pure-C++
// forward, and returns the 9 argmax head indices (region / class_num×3 / hiragana / plate_num×4).
// The page turns those indices into the plate string. All client-side — nothing is uploaded.
#include "net_lpr.hpp"
#include <emscripten/emscripten.h>
#include <vector>
#include <algorithm>

static LProv* g_prov = nullptr;
static int g_out[9];

static void run(const std::vector<float>& chw) {
  LprOut o = lpr_forward(from_data({1, 3, 128, 128}, chw), *g_prov, false);
  for (int h = 0; h < 9; ++h) { auto& d = o.heads[h]->data; int a = 0; float m = -1e9f;
    for (int i = 0; i < (int)d.size(); ++i) if (d[i] > m) { m = d[i]; a = i; } g_out[h] = a; }
}

extern "C" {
EMSCRIPTEN_KEEPALIVE int fn_ready() { if (!g_prov) g_prov = new LProv(load_lpr("/lpr/")); return g_prov ? 1 : 0; }

// rgba: 128*128*4 bytes (already cropped+resized by the page). Returns pointer to 9 argmax ints.
EMSCRIPTEN_KEEPALIVE int* fn_recognize(unsigned char* rgba) {
  if (!g_prov) fn_ready();
  std::vector<float> chw(3 * 128 * 128);
  for (int y = 0; y < 128; ++y) for (int x = 0; x < 128; ++x)
    for (int c = 0; c < 3; ++c) chw[(c * 128 + y) * 128 + x] = rgba[(y * 128 + x) * 4 + c] / 255.f;
  run(chw); return g_out;
}
// pre-standardized NCHW 3x128x128 (node parity test).
EMSCRIPTEN_KEEPALIVE int* fn_recognize_chw(float* chw) {
  if (!g_prov) fn_ready();
  run(std::vector<float>(chw, chw + 3 * 128 * 128)); return g_out;
}
}
