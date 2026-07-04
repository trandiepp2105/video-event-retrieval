from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import EventFormerV1DynamicTSM


class EventAwareMomentLocalizer(nn.Module):
    def __init__(
        self,
        d_raw: int,
        d_model: int = 768,
        text_model_name: str = "roberta-base",
        freeze_text_encoder: bool = True,
        max_frames: int = 2048,
        max_events: int = 512,
        frame_layers: int = 2,
        event_layers: int = 2,
        num_heads: int = 4,
        frame_anchor_sizes: Tuple = (3, 6, 9, "all"),
        event_anchor_sizes: Tuple = (1, 2, 3, "all"),
        ff_dim: int = 3072,
        dropout: float = 0.1,
        event_strategy: str = "tsm",
        event_kmeans_num_events: int = 10,
        event_window_size: int = 5,
        tsm_window_size: int = 4,
        tsm_threshold_alpha: float = 0.5,
        min_event_len: int = 3,
        max_event_len: int = 30,
        use_cross_attention: bool = False,
        use_event_auxiliary_loss: bool = False,
        lambda_event_localizer: float = 0.8,
        freeze_video_encoder_for_localizer: bool = False,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.use_event_auxiliary_loss = use_event_auxiliary_loss
        self.lambda_event_localizer = lambda_event_localizer
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_video_encoder_for_localizer = freeze_video_encoder_for_localizer

        self.video_encoder = EventFormerV1DynamicTSM(
            d_raw=d_raw,
            d_model=d_model,
            text_model_name=text_model_name,
            freeze_text_encoder=freeze_text_encoder,
            query_pooling="mean",
            max_frames=max_frames,
            max_events=max_events,
            frame_layers=frame_layers,
            event_layers=event_layers,
            num_heads=num_heads,
            frame_anchor_sizes=list(frame_anchor_sizes),
            event_anchor_sizes=list(event_anchor_sizes),
            ff_dim=ff_dim,
            dropout=dropout,
            event_strategy=event_strategy,
            event_kmeans_num_events=event_kmeans_num_events,
            event_window_size=event_window_size,
            tsm_window_size=tsm_window_size,
            tsm_threshold_alpha=tsm_threshold_alpha,
            min_event_len=min_event_len,
            max_event_len=max_event_len,
        )
        self.text_encoder = self.video_encoder.text_encoder
        self.text_hidden_size = self.video_encoder.text_hidden_size
        if self.freeze_video_encoder_for_localizer:
            for p in self.video_encoder.parameters():
                p.requires_grad = False

        self.start_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, 1, kernel_size=1),
        )
        self.end_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, 1, kernel_size=1),
        )

        self.query_token_projection = nn.Linear(self.text_hidden_size, d_model)
        self.frame_query_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_dropout = nn.Dropout(dropout)

        self.event_start_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.event_end_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.freeze_text_encoder:
            with torch.no_grad():
                out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        return self.query_token_projection(token_emb)

    @staticmethod
    def _find_event_containing(spans: List[Tuple[int, int]], idx: int) -> int:
        if len(spans) == 0:
            return 0
        for j, (s, e) in enumerate(spans):
            if int(s) <= idx <= int(e):
                return j
        centers = [abs(((int(s) + int(e)) / 2.0) - idx) for s, e in spans]
        return int(min(range(len(centers)), key=lambda k: centers[k]))

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gt_start_idx: Optional[torch.Tensor] = None,
        gt_end_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        video_ctx = torch.no_grad() if self.freeze_video_encoder_for_localizer else nullcontext()
        with video_ctx:
            video_out = self.video_encoder.encode_video_trainable(
                features=features,
                feature_mask=feature_mask,
                normalize=False,
            )
        h = video_out["frame_embeddings"]
        g = video_out["event_embeddings"]
        event_spans = video_out["event_spans"]
        event_mask = video_out["event_mask"]

        if self.use_cross_attention:
            query_tokens = self.encode_text_tokens(input_ids, attention_mask)
            h_cross, _ = self.frame_query_cross_attn(
                query=h,
                key=query_tokens,
                value=query_tokens,
                key_padding_mask=~attention_mask.bool(),
                need_weights=False,
            )
            h = self.cross_norm(h + self.cross_dropout(h_cross))
            h = h * feature_mask.unsqueeze(-1).to(h.dtype)

        x = h.transpose(1, 2)
        start_logits = self.start_head(x).squeeze(1)
        end_logits = self.end_head(x).squeeze(1)
        start_logits = start_logits.masked_fill(~feature_mask.bool(), -1e4)
        end_logits = end_logits.masked_fill(~feature_mask.bool(), -1e4)

        event_start_logits = self.event_start_head(g).squeeze(-1)
        event_end_logits = self.event_end_head(g).squeeze(-1)
        event_start_logits = event_start_logits.masked_fill(~event_mask.bool(), -1e4)
        event_end_logits = event_end_logits.masked_fill(~event_mask.bool(), -1e4)

        loss_frame_loc = None
        loss_event_loc = None
        loss = None
        if gt_start_idx is not None and gt_end_idx is not None:
            loss_start = F.cross_entropy(start_logits, gt_start_idx)
            loss_end = F.cross_entropy(end_logits, gt_end_idx)
            loss_frame_loc = loss_start + loss_end
            loss = loss_frame_loc

            if self.use_event_auxiliary_loss:
                event_start_labels = []
                event_end_labels = []
                for b in range(len(event_spans)):
                    s = int(gt_start_idx[b].item())
                    e = int(gt_end_idx[b].item())
                    event_start_labels.append(self._find_event_containing(event_spans[b], s))
                    event_end_labels.append(self._find_event_containing(event_spans[b], e))
                event_start_labels = torch.tensor(event_start_labels, device=g.device, dtype=torch.long)
                event_end_labels = torch.tensor(event_end_labels, device=g.device, dtype=torch.long)
                loss_event_start = F.cross_entropy(event_start_logits, event_start_labels)
                loss_event_end = F.cross_entropy(event_end_logits, event_end_labels)
                loss_event_loc = loss_event_start + loss_event_end
                loss = loss + self.lambda_event_localizer * loss_event_loc

        return {
            "loss": loss,
            "loss_frame_loc": loss_frame_loc,
            "loss_event_loc": loss_event_loc,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "event_start_logits": event_start_logits,
            "event_end_logits": event_end_logits,
            "event_spans": event_spans,
            "event_mask": event_mask,
        }

    @torch.no_grad()
    def decode_top_spans(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        feature_mask: torch.Tensor,
        topk: int = 5,
        max_span_len: int = 64,
    ) -> List[List[Dict[str, float]]]:
        start_log_prob = torch.log_softmax(start_logits, dim=-1)
        end_log_prob = torch.log_softmax(end_logits, dim=-1)
        outputs: List[List[Dict[str, float]]] = []

        for b in range(start_logits.shape[0]):
            valid_n = int(feature_mask[b].sum().item())
            spans = []
            for s in range(valid_n):
                max_e = min(valid_n - 1, s + max(1, max_span_len) - 1)
                for e in range(s, max_e + 1):
                    score = float((start_log_prob[b, s] + end_log_prob[b, e]).item())
                    spans.append({
                        "start_idx": int(s),
                        "end_idx": int(e),
                        "score": score,
                    })
            spans.sort(key=lambda x: x["score"], reverse=True)
            outputs.append(spans[:topk])
        return outputs
