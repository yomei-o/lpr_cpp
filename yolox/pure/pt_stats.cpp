// Scan a state_dict .pt (pure C++, no Python) and report per-tensor min/max plus any NaN/Inf.
// Localises "detection is NaN" bugs: run on the checkpoint to see WHICH tensor is corrupt.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\pt_stats.cpp   (or g++ / cc.sh)
//   run:   pt_stats <checkpoint.pt>
#include "ptio.hpp"
#include <cstdio>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: pt_stats <checkpoint.pt>\n"); return 1; }
  std::string path = argv[1];
  auto ts = pt::load_pt(path);
  if (ts.empty()) ts = pt::load_pt_module(path);
  if (ts.empty()) { printf("no tensors read from %s\n", path.c_str()); return 1; }
  printf("%zu tensors in %s\n", ts.size(), path.c_str());
  int bad = 0;
  for (auto& t : ts) {
    double mn = 1e300, mx = -1e300; size_t nan = 0, inf = 0;
    for (float v : t.data) {
      if (std::isnan(v)) { ++nan; continue; }
      if (std::isinf(v)) { ++inf; continue; }
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    bool corrupt = nan || inf;
    if (corrupt) ++bad;
    // print corrupt ones always; otherwise only a compact line
    if (corrupt)
      printf("  !! %-40s n=%zu  min=%.3g max=%.3g  NaN=%zu Inf=%zu\n",
             t.name.c_str(), t.data.size(), mn, mx, nan, inf);
  }
  if (bad == 0) printf("clean: no NaN/Inf in any tensor.\n");
  else printf("=> %d tensor(s) contain NaN/Inf.\n", bad);
  // running_var is eval-only: tiny values make eval BN divide by ~sqrt(eps) and amplify
  // activations ~1/sqrt(rv) per layer, which can cascade to Inf->NaN even when every stored
  // value is finite. Report the global minimum + the smallest offenders.
  printf("--- running_var summary (eval-only BN stat; tiny => eval-time blow-up risk) ---\n");
  double grv_min = 1e300; std::string grv_who; size_t nvar = 0, ntiny = 0;
  for (auto& t : ts) if (t.name.size() >= 11 && t.name.substr(t.name.size()-11) == "running_var") {
    ++nvar; double mn = 1e300, mx = -1e300; size_t neg = 0, nan = 0;
    for (float v : t.data) { if (std::isnan(v)) { ++nan; continue; } if (v < 0) ++neg; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mn < grv_min) { grv_min = mn; grv_who = t.name; }
    if (nan || neg || mn < 1e-4) { ++ntiny; printf("  %-44s min=%.3g max=%.3g neg=%zu NaN=%zu\n", t.name.c_str(), mn, mx, neg, nan); }
  }
  printf("running_var tensors=%zu  global min=%.3g (%s)  tiny(<1e-4)=%zu\n",
         nvar, grv_min, grv_who.c_str(), ntiny);
  return bad ? 2 : 0;
}
