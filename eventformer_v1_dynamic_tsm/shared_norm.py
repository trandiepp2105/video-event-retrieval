from __future__ import annotations

import torch
import torch.nn.functional as F


def shared_norm_start_end_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_candidate_idx: torch.Tensor,
    gt_start_idx: torch.Tensor,
    gt_end_idx: torch.Tensor,
):
    bsz, num_candidates, seq_len = start_logits.shape

    start_flat = start_logits.reshape(bsz, num_candidates * seq_len)
    end_flat = end_logits.reshape(bsz, num_candidates * seq_len)
    mask_flat = candidate_mask.reshape(bsz, num_candidates * seq_len).bool()

    start_flat = start_flat.masked_fill(~mask_flat, -1e4)
    end_flat = end_flat.masked_fill(~mask_flat, -1e4)

    target_start = positive_candidate_idx.long() * seq_len + gt_start_idx.long()
    target_end = positive_candidate_idx.long() * seq_len + gt_end_idx.long()

    loss_start = F.cross_entropy(start_flat, target_start)
    loss_end = F.cross_entropy(end_flat, target_end)
    return loss_start + loss_end
