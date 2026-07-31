// Multi-head loss for the LPR classifier: sum of per-head softmax cross-entropy over the 9 heads
// (region, class_num×3, hiragana, plate_num×4). Uses the shared cross_entropy (log-softmax + NLL).
#pragma once
#include "net_lpr.hpp"
#include "face_loss.hpp"      // cross_entropy(logits, labels)

// out.heads = 9 logit tensors [B,C]; labels[h] = int label per sample (length B).
inline Tensor lpr_loss(const LprOut& out, const std::vector<std::vector<int>>& labels) {
  Tensor total = cross_entropy(out.heads[0], labels[0]);
  for (int h = 1; h < 9; ++h) total = add(total, cross_entropy(out.heads[h], labels[h]));
  return total;
}
