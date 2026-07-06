
# eventformer_v1_dynamic_tsm_model.py
# Shared model module for EventFormer V1 dynamic TSM notebooks.

import math
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import resolve_text_model_source
from .event_reasoning import EventReasoner


def pool_events(
    frame_embeddings: torch.Tensor,
    event_spans_batch: List[List[Tuple[int, int]]],
    max_events: int,
    pooling: str = "max",
):
    B, _, D = frame_embeddings.shape
    if len(event_spans_batch) == 0:
        return frame_embeddings.new_zeros(B, 0, D), torch.zeros(B, 0, dtype=torch.bool, device=frame_embeddings.device)

    max_events_in_batch = max((len(spans) for spans in event_spans_batch), default=0)
    M = min(max_events, max_events_in_batch) if max_events > 0 else max_events_in_batch
    if M <= 0:
        return frame_embeddings.new_zeros(B, 0, D), torch.zeros(B, 0, dtype=torch.bool, device=frame_embeddings.device)

    event_embeddings = frame_embeddings.new_zeros(B, M, D)
    event_mask = torch.zeros(B, M, dtype=torch.bool, device=frame_embeddings.device)

    for b, spans in enumerate(event_spans_batch):
        for m, (s, e) in enumerate(spans[:M]):
            x = frame_embeddings[b, s : e + 1]
            if x.numel() == 0:
                continue
            if pooling == "max":
                emb = x.max(dim=0).values
            elif pooling == "mean":
                emb = x.mean(dim=0)
            else:
                raise ValueError(f"Unknown event_pooling: {pooling}")
            event_embeddings[b, m] = emb
            event_mask[b, m] = True

    return event_embeddings, event_mask


def find_event_containing_frame(event_spans: List[Tuple[int, int]], frame_idx: int) -> Optional[int]:
    for idx, (s, e) in enumerate(event_spans):
        if s <= frame_idx <= e:
            return idx
    return None


def find_best_iou_event(event_spans: List[Tuple[int, int]], gt_start: int, gt_end: int) -> Tuple[Optional[int], float]:
    if len(event_spans) == 0:
        return None, -1.0
    gt = (int(gt_start), int(gt_end))
    best_idx = None
    best_iou = -1.0
    for idx, span in enumerate(event_spans):
        inter = max(0, min(span[1], gt[1]) - max(span[0], gt[0]) + 1)
        union = (span[1] - span[0] + 1) + (gt[1] - gt[0] + 1) - inter
        iou = 0.0 if union <= 0 else inter / union
        if iou > best_iou:
            best_idx = idx
            best_iou = iou
    return best_idx, best_iou


def _spans_to_tensor(
    event_spans_batch: List[List[Tuple[int, int]]],
    max_events: int,
    device: torch.device,
):
    batch_size = len(event_spans_batch)
    span_tensor = torch.zeros(batch_size, max_events, 2, dtype=torch.long, device=device)
    span_mask = torch.zeros(batch_size, max_events, dtype=torch.bool, device=device)
    for b, spans in enumerate(event_spans_batch):
        clipped = spans[:max_events]
        if not clipped:
            continue
        span_tensor[b, : len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=device)
        span_mask[b, : len(clipped)] = True
    return span_tensor, span_mask


class AnchorMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with per-head temporal anchor windows.

    anchor_sizes examples:
      [3, 6, 9, "all"] means head 0 attends within +/-3 positions,
      head 1 within +/-6, head 2 within +/-9, head 3 globally.
    """
    def __init__(self, d_model: int, num_heads: int, anchor_sizes: List[Any], dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        if len(anchor_sizes) < num_heads:
            anchor_sizes = list(anchor_sizes) + ["all"] * (num_heads - len(anchor_sizes))
        self.anchor_sizes = anchor_sizes[:num_heads]
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def _build_anchor_mask(self, batch_size: int, seq_len: int, valid_mask: Optional[torch.Tensor], device):
        # mask shape required by nn.MultiheadAttention: [B * num_heads, L, L]
        idx = torch.arange(seq_len, device=device)
        distance = (idx[:, None] - idx[None, :]).abs()
        per_head = []
        for anchor in self.anchor_sizes:
            if anchor == "all" or anchor is None:
                m = torch.zeros(seq_len, seq_len, device=device)
            else:
                radius = int(anchor)
                m = torch.zeros(seq_len, seq_len, device=device)
                m = m.masked_fill(distance > radius, float("-inf"))
            per_head.append(m)
        base = torch.stack(per_head, dim=0)  # [H, L, L]
        mask = base.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, L, L]

        if valid_mask is not None:
            # Mask invalid keys for valid queries.
            invalid = ~valid_mask.bool()
            invalid_keys = invalid.view(batch_size, 1, 1, seq_len)
            mask = mask.masked_fill(invalid_keys, float("-inf"))
            # Invalid padded queries may have no valid key in a local window, which can create NaNs.
            # They will be zeroed after attention, so let them attend anywhere here.
            invalid_queries = invalid.view(batch_size, 1, seq_len, 1)
            mask = mask.masked_fill(invalid_queries, 0.0)

        mask = mask.reshape(batch_size * self.num_heads, seq_len, seq_len)
        return mask

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        attn_mask = self._build_anchor_mask(B, L, valid_mask, x.device)
        y, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        if valid_mask is not None:
            y = y * valid_mask.unsqueeze(-1).to(y.dtype)
        return y


class AnchorFormerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, anchor_sizes: List[Any], ff_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AnchorMultiHeadSelfAttention(d_model, num_heads, anchor_sizes, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        x = x + self.drop1(self.attn(self.norm1(x), valid_mask))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x


class AnchorFormerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, anchor_sizes: List[Any], ff_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AnchorFormerLayer(d_model, num_heads, anchor_sizes, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x = layer(x, valid_mask)
        x = self.norm(x)
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        return x


class EventFormerV1DynamicTSM(nn.Module):
    PAPER_QUERY_POOLING = "attention"
    PAPER_EVENT_STRATEGY = "contrastive_convolution"
    PAPER_EVENT_POOLING = "max"
    PAPER_MODALITIES = ("visual",)
    PAPER_QUERY_TRANSFORMER_LAYERS = 1
    PAPER_DROPOUT = 0.1
    PAPER_LAMBDA_FRAME = 1.0
    PAPER_LAMBDA_EVENT = 0.8
    PAPER_WEAK_POSITIVE_WEIGHT = 0.5
    PAPER_TEMPERATURE = 0.01

    def __init__(
        self,
        d_raw: int,
        d_model: int = 768,
        text_model_name: str = "roberta-base",
        text_model_path: Optional[str] = None,
        freeze_text_encoder: bool = True,
        query_pooling: str = "attention",
        query_transformer_layers: int = 1,
        query_transformer_heads: Optional[int] = None,
        query_transformer_ff_dim: Optional[int] = None,
        query_transformer_dropout: Optional[float] = None,
        use_modality_specific_query: bool = False,
        modalities: Tuple[str, ...] = ("visual",),
        max_frames: int = 2048,
        max_events: int = 512,
        frame_layers: int = 2,
        event_layers: int = 2,
        num_heads: int = 4,
        frame_anchor_sizes: List[Any] = (3, 6, 9, "all"),
        event_anchor_sizes: List[Any] = (1, 2, 3, "all"),
        ff_dim: int = 3072,
        dropout: float = 0.1,
        event_strategy: str = "contrastive_convolution",
        event_kmeans_num_events: int = 10,
        event_window_size: int = 8,
        event_stride: Optional[int] = None,
        event_window_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64),
        event_stride_ratio: float = 0.5,
        event_pooling: str = "max",
        tsm_window_size: int = 4,
        tsm_threshold_alpha: float = 0.5,
        min_event_len: int = 3,
        max_event_len: int = 30,
        normalize_embeddings: bool = True,
        lambda_frame: float = 1.0,
        lambda_event: float = 0.8,
        weak_positive_weight: float = 0.5,
        use_hard_negative: bool = True,
        lambda_hard: float = 1.0,
        use_weak_positive: bool = True,
        lambda_weak: float = 0.1,
        lambda_weak_event: Optional[float] = None,
        weak_positive_margin: int = 10,
        temperature: float = 0.01,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is not installed. Install it before creating the model.")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got d_model={d_model}, num_heads={num_heads}")
        paper_query_pooling = self.PAPER_QUERY_POOLING
        paper_event_strategy = self.PAPER_EVENT_STRATEGY
        paper_event_pooling = self.PAPER_EVENT_POOLING
        paper_modalities = self.PAPER_MODALITIES
        paper_query_transformer_layers = self.PAPER_QUERY_TRANSFORMER_LAYERS
        paper_dropout = self.PAPER_DROPOUT
        paper_lambda_frame = self.PAPER_LAMBDA_FRAME
        paper_lambda_event = self.PAPER_LAMBDA_EVENT
        paper_weak_positive_weight = self.PAPER_WEAK_POSITIVE_WEIGHT
        paper_temperature = self.PAPER_TEMPERATURE

        self.config_dict = dict(
            d_raw=d_raw,
            d_model=d_model,
            text_model_name=text_model_name,
            text_model_path=text_model_path,
            freeze_text_encoder=freeze_text_encoder,
            query_pooling=paper_query_pooling,
            query_transformer_layers=paper_query_transformer_layers,
            query_transformer_heads=query_transformer_heads,
            query_transformer_ff_dim=query_transformer_ff_dim,
            query_transformer_dropout=query_transformer_dropout,
            use_modality_specific_query=False,
            modalities=list(paper_modalities),
            max_frames=max_frames,
            max_events=max_events,
            frame_layers=frame_layers,
            event_layers=event_layers,
            num_heads=num_heads,
            frame_anchor_sizes=list(frame_anchor_sizes),
            event_anchor_sizes=list(event_anchor_sizes),
            ff_dim=ff_dim,
            dropout=paper_dropout,
            event_strategy=paper_event_strategy,
            event_kmeans_num_events=event_kmeans_num_events,
            event_window_size=event_window_size,
            event_stride=event_stride,
            event_window_sizes=tuple(event_window_sizes),
            event_stride_ratio=event_stride_ratio,
            event_pooling=paper_event_pooling,
            tsm_window_size=tsm_window_size,
            tsm_threshold_alpha=tsm_threshold_alpha,
            min_event_len=min_event_len,
            max_event_len=max_event_len,
            normalize_embeddings=normalize_embeddings,
            lambda_frame=paper_lambda_frame,
            lambda_event=paper_lambda_event,
            weak_positive_weight=paper_weak_positive_weight,
            use_hard_negative=True,
            lambda_hard=lambda_hard,
            use_weak_positive=True,
            lambda_weak=lambda_weak,
            lambda_weak_event=lambda_weak if lambda_weak_event is None else lambda_weak_event,
            weak_positive_margin=weak_positive_margin,
            temperature=paper_temperature,
        )
        self.d_model = d_model
        self.normalize_embeddings = normalize_embeddings
        self.lambda_frame = paper_lambda_frame
        self.lambda_event = paper_lambda_event
        self.weak_positive_weight = paper_weak_positive_weight
        self.use_hard_negative = True
        self.lambda_hard = lambda_hard
        self.use_weak_positive = True
        self.lambda_weak = lambda_weak
        self.lambda_weak_event = lambda_weak if lambda_weak_event is None else lambda_weak_event
        self.weak_positive_margin = weak_positive_margin
        self.temperature = paper_temperature
        self.tsm_window_size = tsm_window_size
        self.tsm_threshold_alpha = tsm_threshold_alpha
        self.min_event_len = min_event_len
        self.max_event_len = max_event_len
        self.max_frames = max_frames
        self.max_events = max_events
        self.event_strategy = paper_event_strategy
        self.event_pooling = paper_event_pooling
        self.freeze_text_encoder = freeze_text_encoder
        self.query_pooling = paper_query_pooling
        self.query_transformer_layers = paper_query_transformer_layers
        self.use_modality_specific_query = False
        self.modalities = paper_modalities

        self.text_model_name = text_model_name
        self.text_model_path = text_model_path
        self.text_model_source, self.text_model_local_only = resolve_text_model_source(text_model_name, text_model_path)

        self.visual_projection = nn.Linear(d_raw, d_model)
        self.frame_pos_embed = nn.Embedding(max_frames, d_model)
        self.frame_encoder = AnchorFormerEncoder(frame_layers, d_model, num_heads, list(frame_anchor_sizes), ff_dim, paper_dropout)

        self.event_pos_embed = nn.Embedding(max_events, d_model)
        self.event_encoder = AnchorFormerEncoder(event_layers, d_model, num_heads, list(event_anchor_sizes), ff_dim, paper_dropout)
        self.event_reasoner = EventReasoner(
            strategy=paper_event_strategy,
            tsm_window_size=tsm_window_size,
            tsm_threshold_alpha=tsm_threshold_alpha,
            min_event_len=min_event_len,
            max_event_len=max_event_len,
            kmeans_num_events=event_kmeans_num_events,
            window_size=event_window_size,
            stride=event_stride,
            window_sizes=tuple(event_window_sizes),
            stride_ratio=event_stride_ratio,
            max_events=max_events,
        )

        self.text_encoder = AutoModel.from_pretrained(
            self.text_model_source,
            local_files_only=self.text_model_local_only,
        )
        self.text_hidden_size = int(self.text_encoder.config.hidden_size)
        text_dim = self.text_hidden_size
        self.query_transformer_heads = int(query_transformer_heads or num_heads)
        if text_dim % self.query_transformer_heads != 0:
            raise ValueError(
                "text hidden size must be divisible by query_transformer_heads, "
                f"got hidden_size={text_dim}, query_transformer_heads={self.query_transformer_heads}"
            )
        self.query_transformer_ff_dim = int(query_transformer_ff_dim or text_dim * 4)
        self.query_transformer_dropout = float(paper_dropout if query_transformer_dropout is None else query_transformer_dropout)
        if self.query_transformer_layers > 0:
            query_layer = nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=self.query_transformer_heads,
                dim_feedforward=self.query_transformer_ff_dim,
                dropout=self.query_transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.query_transformer = nn.TransformerEncoder(
                query_layer,
                num_layers=self.query_transformer_layers,
                norm=nn.LayerNorm(text_dim),
            )
        else:
            self.query_transformer = nn.Identity()
        self.query_projection = nn.Linear(text_dim, d_model)
        self.query_attn_pool = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.Tanh(),
            nn.Linear(text_dim, 1),
        )
        self.modality_query_pools = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.Tanh(),
                nn.Linear(text_dim, 1),
            )
            for m in self.modalities
        })
        self.modality_query_projections = nn.ModuleDict({
            m: nn.Linear(text_dim, d_model)
            for m in self.modalities
        })

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def _pool_query_tokens(self, token_emb: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.bool()
        scores = self.query_attn_pool(token_emb).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(token_emb * weights.unsqueeze(-1), dim=1)
        return pooled

    def _encode_query_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.freeze_text_encoder:
            with torch.no_grad():
                out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        if self.query_transformer_layers > 0:
            token_emb = self.query_transformer(
                token_emb,
                src_key_padding_mask=~attention_mask.bool(),
            )
            token_emb = token_emb * attention_mask.unsqueeze(-1).to(token_emb.dtype)
        return token_emb

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        token_emb = self._encode_query_tokens(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_query_tokens(token_emb, attention_mask)
        q = self.query_projection(pooled)
        return q

    def encode_text_multi(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        token_emb = self._encode_query_tokens(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.bool()
        query_dict = {}
        for m in self.modalities:
            scores = self.modality_query_pools[m](token_emb).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.sum(token_emb * weights.unsqueeze(-1), dim=1)
            query_dict[m] = self.modality_query_projections[m](pooled)
        return query_dict

    def _valid_positions(self, length: int, max_len: int, device):
        pos = torch.arange(length, device=device)
        pos = torch.clamp(pos, max=max_len - 1)
        return pos

    def _project_and_encode_frames(self, features: torch.Tensor, feature_mask: torch.Tensor):
        B, N, _ = features.shape
        x = self.visual_projection(features)
        pos = self._valid_positions(N, self.max_frames, features.device)
        x = x + self.frame_pos_embed(pos).unsqueeze(0)
        x = x * feature_mask.unsqueeze(-1).to(x.dtype)
        h = self.frame_encoder(x, feature_mask)
        return h

    def encode_video_batch(self, features: torch.Tensor, feature_mask: torch.Tensor, return_spans: bool = True):
        B, N, _ = features.shape
        frame_embeddings = self._project_and_encode_frames(features, feature_mask)
        event_spans_batch: List[List[Tuple[int, int]]] = []
        for b in range(B):
            valid_len = int(feature_mask[b].sum().item()) if feature_mask is not None else N
            spans = self.event_reasoner.detect_event_spans(frame_embeddings[b, :valid_len])
            event_spans_batch.append(spans[: self.max_events])

        initial_event_embeddings, event_mask = pool_events(
            frame_embeddings=frame_embeddings,
            event_spans_batch=event_spans_batch,
            max_events=self.max_events,
            pooling=self.event_pooling,
        )
        if initial_event_embeddings.shape[1] == 0:
            initial_event_embeddings = frame_embeddings.new_zeros(B, 1, self.d_model)
            event_mask = torch.zeros(B, 1, dtype=torch.bool, device=frame_embeddings.device)

        M = initial_event_embeddings.shape[1]
        pos = self._valid_positions(M, self.max_events, frame_embeddings.device)
        event_x = initial_event_embeddings + self.event_pos_embed(pos).unsqueeze(0)
        event_x = event_x * event_mask.unsqueeze(-1).to(event_x.dtype)
        event_embeddings = self.event_encoder(event_x, event_mask)
        return frame_embeddings, event_embeddings, event_mask, event_spans_batch

    def compute_retrieval_scores(
        self,
        query_embedding: torch.Tensor,
        frame_embeddings: torch.Tensor,
        event_embeddings: torch.Tensor,
        feature_mask: torch.Tensor,
        event_mask: torch.Tensor,
    ):
        q_norm = F.normalize(query_embedding, dim=-1) if self.normalize_embeddings else query_embedding
        h_norm = F.normalize(frame_embeddings, dim=-1) if self.normalize_embeddings else frame_embeddings
        g_norm = F.normalize(event_embeddings, dim=-1) if self.normalize_embeddings else event_embeddings

        neg_inf = torch.finfo(q_norm.dtype).min
        frame_scores = torch.einsum("bd,bnd->bn", q_norm, h_norm).masked_fill(~feature_mask.bool(), neg_inf)
        event_scores = torch.einsum("bd,bmd->bm", q_norm, g_norm).masked_fill(~event_mask.bool(), neg_inf)

        frame_video_scores = frame_scores.max(dim=1).values
        event_video_scores = event_scores.max(dim=1).values
        video_scores = torch.where(event_mask.any(dim=1), event_video_scores, frame_video_scores)
        return {
            "frame_scores": frame_scores,
            "event_scores": event_scores,
            "frame_video_scores": frame_video_scores,
            "event_video_scores": event_video_scores,
            "video_scores": video_scores,
        }

    def _build_gt_mask(
        self,
        seq_len: int,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ):
        device = gt_start_idx.device
        if valid_mask is None:
            valid_mask = torch.ones(gt_start_idx.shape[0], seq_len, dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.bool()

        valid_len = valid_mask.sum(dim=1).clamp(min=1)
        max_idx = valid_len - 1
        gt_start = torch.minimum(gt_start_idx.long().clamp(min=0), max_idx)
        gt_end = torch.minimum(gt_end_idx.long().clamp(min=0), max_idx)
        gt_end = torch.maximum(gt_end, gt_start)

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        gt_mask = (positions >= gt_start.unsqueeze(1)) & (positions <= gt_end.unsqueeze(1)) & valid_mask
        return gt_mask, gt_start, gt_end, valid_mask

    def _mine_positive_frames(
        self,
        query_embedding: torch.Tensor,
        frame_embeddings: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        feature_mask: Optional[torch.Tensor],
    ):
        q = F.normalize(query_embedding, dim=-1)
        frames = F.normalize(frame_embeddings, dim=-1)
        frame_scores = torch.einsum("bd,bnd->bn", q, frames)
        gt_mask, gt_start, gt_end, valid_mask = self._build_gt_mask(
            seq_len=frame_embeddings.shape[1],
            gt_start_idx=gt_start_idx,
            gt_end_idx=gt_end_idx,
            valid_mask=feature_mask,
        )

        neg_inf = torch.finfo(frame_scores.dtype).min
        masked_gt_scores = frame_scores.masked_fill(~gt_mask, neg_inf)
        pos_frame_idx = masked_gt_scores.argmax(dim=1)
        batch_idx = torch.arange(query_embedding.shape[0], device=query_embedding.device)
        pos_frames = frame_embeddings[batch_idx, pos_frame_idx]

        outside_mask = valid_mask & ~gt_mask
        masked_outside_scores = frame_scores.masked_fill(~outside_mask, neg_inf)
        weak_frame_idx = masked_outside_scores.argmax(dim=1)
        weak_frames = frame_embeddings[batch_idx, weak_frame_idx]
        weak_valid_mask = outside_mask.any(dim=1)

        return {
            "q_norm": q,
            "frames_norm": frames,
            "frame_scores": frame_scores,
            "gt_mask": gt_mask,
            "valid_mask": valid_mask,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "pos_frame_idx": pos_frame_idx,
            "pos_frames": pos_frames,
            "weak_frame_idx": weak_frame_idx,
            "weak_frames": weak_frames,
            "weak_valid_mask": weak_valid_mask,
        }

    def _mine_positive_events(
        self,
        query_embedding: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_spans_batch: List[List[Tuple[int, int]]],
        event_mask: torch.Tensor,
        pos_frame_idx: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
    ):
        batch_size, max_events, _ = event_embeddings.shape
        span_tensor, span_valid_mask = _spans_to_tensor(
            event_spans_batch=event_spans_batch,
            max_events=max_events,
            device=event_embeddings.device,
        )
        valid_event_mask = event_mask.bool() & span_valid_mask
        batch_idx = torch.arange(batch_size, device=event_embeddings.device)

        starts = span_tensor[:, :, 0]
        ends = span_tensor[:, :, 1]
        frame_idx = pos_frame_idx.unsqueeze(1)
        contains_mask = valid_event_mask & (starts <= frame_idx) & (frame_idx <= ends)
        contains_any = contains_mask.any(dim=1)
        contains_idx = contains_mask.to(torch.int64).argmax(dim=1)

        gt_start = gt_start_idx.long().unsqueeze(1)
        gt_end = gt_end_idx.long().unsqueeze(1)
        inter = (torch.minimum(ends, gt_end) - torch.maximum(starts, gt_start) + 1).clamp(min=0)
        span_len = (ends - starts + 1).clamp(min=0)
        gt_len = (gt_end - gt_start + 1).clamp(min=1)
        union = span_len + gt_len - inter
        iou = inter.to(event_embeddings.dtype) / union.clamp(min=1).to(event_embeddings.dtype)
        iou = iou.masked_fill(~valid_event_mask, -1.0)
        best_iou_idx = iou.argmax(dim=1)

        pos_event_idx = torch.where(contains_any, contains_idx, best_iou_idx)
        valid_positive = valid_event_mask.any(dim=1)
        pos_events = event_embeddings[batch_idx, pos_event_idx]

        return {
            "pos_event_idx": pos_event_idx,
            "pos_events": pos_events,
            "valid_positive": valid_positive,
            "span_tensor": span_tensor,
            "span_mask": valid_event_mask,
        }

    def _mine_hard_negative_scores_per_video(
        self,
        query_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
    ):
        q = F.normalize(query_embedding, dim=-1)
        candidates = F.normalize(candidate_embeddings, dim=-1)
        batch_size = q.shape[0]
        neg_inf = torch.finfo(q.dtype).min
        hardest_scores = q.new_full((batch_size, batch_size), neg_inf)

        for video_idx in range(batch_size):
            valid_local = candidate_mask[video_idx].bool()
            if not valid_local.any():
                continue
            scores = q @ candidates[video_idx, valid_local].transpose(0, 1)
            hardest_scores[:, video_idx] = scores.max(dim=1).values
        return hardest_scores

    def _query_to_candidates_info_nce(
        self,
        pos_scores: torch.Tensor,
        neg_scores_by_video: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = pos_scores.shape[0]
        neg_inf = torch.finfo(pos_scores.dtype).min
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=pos_scores.device)
        neg_scores = neg_scores_by_video.masked_fill(self_mask, neg_inf)
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1) / self.temperature
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if not valid_mask.any():
                return pos_scores.sum() * 0.0, logits
            return F.cross_entropy(logits[valid_mask], targets[valid_mask]), logits
        return F.cross_entropy(logits, targets), logits

    def _candidate_to_query_info_nce(
        self,
        query_embedding: torch.Tensor,
        positive_embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        q = F.normalize(query_embedding, dim=-1)
        positives = F.normalize(positive_embeddings, dim=-1)
        logits = (positives @ q.t()) / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if not valid_mask.any():
                return query_embedding.sum() * 0.0, logits
            return F.cross_entropy(logits[valid_mask], targets[valid_mask]), logits
        return F.cross_entropy(logits, targets), logits

    def compute_frame_contrastive_loss(
        self,
        query_embedding: torch.Tensor,
        frame_embeddings: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ):
        mined = self._mine_positive_frames(
            query_embedding=query_embedding,
            frame_embeddings=frame_embeddings,
            gt_start_idx=gt_start_idx,
            gt_end_idx=gt_end_idx,
            feature_mask=feature_mask,
        )
        valid_positive = mined["gt_mask"].any(dim=1)
        batch_idx = torch.arange(query_embedding.shape[0], device=query_embedding.device)
        pos_scores = mined["frame_scores"][batch_idx, mined["pos_frame_idx"]]
        hard_neg_scores = self._mine_hard_negative_scores_per_video(
            query_embedding=query_embedding,
            candidate_embeddings=frame_embeddings,
            candidate_mask=mined["valid_mask"],
        )
        loss_q2f, _ = self._query_to_candidates_info_nce(
            pos_scores=pos_scores,
            neg_scores_by_video=hard_neg_scores,
            valid_mask=valid_positive,
        )
        loss_f2q, _ = self._candidate_to_query_info_nce(
            query_embedding=query_embedding,
            positive_embeddings=mined["pos_frames"],
            valid_mask=valid_positive,
        )
        loss_frame = loss_q2f + loss_f2q

        if self.use_weak_positive and self.weak_positive_weight > 0:
            weak_scores = mined["frame_scores"][batch_idx, mined["weak_frame_idx"]]
            loss_weak, _ = self._query_to_candidates_info_nce(
                pos_scores=weak_scores,
                neg_scores_by_video=hard_neg_scores.detach(),
                valid_mask=mined["weak_valid_mask"],
            )
            loss_frame = loss_frame + (self.weak_positive_weight * loss_weak)
        return loss_frame

    def compute_event_contrastive_loss(
        self,
        query_embedding: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_spans_batch: List[List[Tuple[int, int]]],
        event_mask: torch.Tensor,
        frame_embeddings: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ):
        mined_frames = self._mine_positive_frames(
            query_embedding=query_embedding,
            frame_embeddings=frame_embeddings,
            gt_start_idx=gt_start_idx,
            gt_end_idx=gt_end_idx,
            feature_mask=feature_mask,
        )
        mined_events = self._mine_positive_events(
            query_embedding=query_embedding,
            event_embeddings=event_embeddings,
            event_spans_batch=event_spans_batch,
            event_mask=event_mask,
            pos_frame_idx=mined_frames["pos_frame_idx"],
            gt_start_idx=mined_frames["gt_start"],
            gt_end_idx=mined_frames["gt_end"],
        )
        batch_idx = torch.arange(query_embedding.shape[0], device=query_embedding.device)
        event_scores = torch.einsum(
            "bd,bmd->bm",
            F.normalize(query_embedding, dim=-1),
            F.normalize(event_embeddings, dim=-1),
        ).masked_fill(~event_mask.bool(), torch.finfo(event_embeddings.dtype).min)
        pos_event_scores = event_scores[batch_idx, mined_events["pos_event_idx"]]
        hard_neg_event_scores = self._mine_hard_negative_scores_per_video(
            query_embedding=query_embedding,
            candidate_embeddings=event_embeddings,
            candidate_mask=event_mask,
        )
        loss_q2e, _ = self._query_to_candidates_info_nce(
            pos_scores=pos_event_scores,
            neg_scores_by_video=hard_neg_event_scores,
            valid_mask=mined_events["valid_positive"],
        )
        loss_e2q, _ = self._candidate_to_query_info_nce(
            query_embedding=query_embedding,
            positive_embeddings=mined_events["pos_events"],
            valid_mask=mined_events["valid_positive"],
        )
        return loss_q2e + loss_e2q

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gt_start_idx: Optional[torch.Tensor],
        gt_end_idx: Optional[torch.Tensor],
        video_ids=None,
    ):
        q = self.encode_text(input_ids, attention_mask)
        h, g, event_mask, all_spans = self.encode_video_batch(features, feature_mask, return_spans=True)

        retrieval_scores = self.compute_retrieval_scores(
            query_embedding=q,
            frame_embeddings=h,
            event_embeddings=g,
            feature_mask=feature_mask,
            event_mask=event_mask,
        )

        out = {
            "query_embedding": q,
            "frame_embeddings": h,
            "event_embeddings": g,
            "event_spans": all_spans,
            "event_mask": event_mask,
            **retrieval_scores,
        }

        if gt_start_idx is not None and gt_end_idx is not None:
            loss_frame = self.compute_frame_contrastive_loss(
                query_embedding=q,
                frame_embeddings=h,
                gt_start_idx=gt_start_idx,
                gt_end_idx=gt_end_idx,
                feature_mask=feature_mask,
            )
            loss_event = self.compute_event_contrastive_loss(
                query_embedding=q,
                event_embeddings=g,
                event_spans_batch=all_spans,
                event_mask=event_mask,
                frame_embeddings=h,
                gt_start_idx=gt_start_idx,
                gt_end_idx=gt_end_idx,
                feature_mask=feature_mask,
            )
            out.update(
                {
                    "loss": (self.lambda_frame * loss_frame) + (self.lambda_event * loss_event),
                    "loss_frame": loss_frame.detach(),
                    "loss_event": loss_event.detach(),
                }
            )
        return out

    @torch.inference_mode()
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: Optional[bool] = None, modality: str = "visual"):
        q = self.encode_text(input_ids, attention_mask)
        if normalize is None:
            normalize = self.normalize_embeddings
        if normalize:
            q = F.normalize(q, dim=-1)
        return q

    @torch.inference_mode()
    def encode_video(self, features: torch.Tensor, feature_mask: torch.Tensor, normalize: Optional[bool] = None):
        return self.encode_video_trainable(features=features, feature_mask=feature_mask, normalize=normalize)

    def encode_video_trainable(self, features: torch.Tensor, feature_mask: torch.Tensor, normalize: Optional[bool] = None):
        h, g, event_mask, all_spans = self.encode_video_batch(features, feature_mask, return_spans=True)
        if normalize is None:
            normalize = self.normalize_embeddings
        if normalize:
            h = F.normalize(h, dim=-1)
            g = F.normalize(g, dim=-1)
        return {
            "frame_embeddings": h,
            "feature_mask": feature_mask,
            "frame_mask": feature_mask,
            "event_embeddings": g,
            "event_mask": event_mask,
            "event_spans": all_spans,
        }
