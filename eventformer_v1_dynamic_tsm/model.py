
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
    def __init__(
        self,
        d_raw: int,
        d_model: int = 768,
        text_model_name: str = "roberta-base",
        text_model_path: Optional[str] = None,
        freeze_text_encoder: bool = True,
        query_pooling: str = "attention",
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
        event_strategy: str = "window",
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
        lambda_frame: float = 0.8,
        lambda_event: float = 1.0,
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
        self.config_dict = dict(
            d_raw=d_raw,
            d_model=d_model,
            text_model_name=text_model_name,
            text_model_path=text_model_path,
            freeze_text_encoder=freeze_text_encoder,
            query_pooling=query_pooling,
            use_modality_specific_query=use_modality_specific_query,
            modalities=list(modalities),
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
            event_stride=event_stride,
            event_window_sizes=tuple(event_window_sizes),
            event_stride_ratio=event_stride_ratio,
            event_pooling=event_pooling,
            tsm_window_size=tsm_window_size,
            tsm_threshold_alpha=tsm_threshold_alpha,
            min_event_len=min_event_len,
            max_event_len=max_event_len,
            normalize_embeddings=normalize_embeddings,
            lambda_frame=lambda_frame,
            lambda_event=lambda_event,
            weak_positive_weight=weak_positive_weight,
            use_hard_negative=use_hard_negative,
            lambda_hard=lambda_hard,
            use_weak_positive=use_weak_positive,
            lambda_weak=lambda_weak,
            lambda_weak_event=lambda_weak if lambda_weak_event is None else lambda_weak_event,
            weak_positive_margin=weak_positive_margin,
            temperature=temperature,
        )
        self.d_model = d_model
        self.normalize_embeddings = normalize_embeddings
        self.lambda_frame = lambda_frame
        self.lambda_event = lambda_event
        self.weak_positive_weight = weak_positive_weight
        self.use_hard_negative = use_hard_negative
        self.lambda_hard = lambda_hard
        self.use_weak_positive = use_weak_positive
        self.lambda_weak = lambda_weak
        self.lambda_weak_event = lambda_weak if lambda_weak_event is None else lambda_weak_event
        self.weak_positive_margin = weak_positive_margin
        self.temperature = temperature
        self.tsm_window_size = tsm_window_size
        self.tsm_threshold_alpha = tsm_threshold_alpha
        self.min_event_len = min_event_len
        self.max_event_len = max_event_len
        self.max_frames = max_frames
        self.max_events = max_events
        self.event_strategy = event_strategy
        self.event_pooling = event_pooling
        self.freeze_text_encoder = freeze_text_encoder
        self.query_pooling = query_pooling
        self.use_modality_specific_query = use_modality_specific_query
        self.modalities = tuple(modalities)
        if self.use_modality_specific_query and "visual" not in self.modalities:
            raise ValueError(
                "modalities must include 'visual' when use_modality_specific_query=True, "
                "because the current visual-only retriever uses q_dict['visual'] to match "
                "query embeddings with ViT+SlowFast visual features."
            )

        self.text_model_name = text_model_name
        self.text_model_path = text_model_path
        self.text_model_source, self.text_model_local_only = resolve_text_model_source(text_model_name, text_model_path)

        self.visual_projection = nn.Linear(d_raw, d_model)
        self.frame_pos_embed = nn.Embedding(max_frames, d_model)
        self.frame_encoder = AnchorFormerEncoder(frame_layers, d_model, num_heads, list(frame_anchor_sizes), ff_dim, dropout)

        self.event_pos_embed = nn.Embedding(max_events, d_model)
        self.event_encoder = AnchorFormerEncoder(event_layers, d_model, num_heads, list(event_anchor_sizes), ff_dim, dropout)
        self.event_reasoner = EventReasoner(
            strategy=event_strategy,
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

        self.logit_scale_frame = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        self.logit_scale_event = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def _pool_query_tokens(self, token_emb: torch.Tensor, attention_mask: torch.Tensor):
        if self.query_pooling == "cls":
            return token_emb[:, 0]

        mask = attention_mask.bool()

        if self.query_pooling == "mean":
            mask_f = mask.unsqueeze(-1).to(token_emb.dtype)
            pooled = (token_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            return pooled

        if self.query_pooling == "attention":
            scores = self.query_attn_pool(token_emb).squeeze(-1)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.sum(token_emb * weights.unsqueeze(-1), dim=1)
            return pooled

        raise ValueError(f"Unknown query_pooling: {self.query_pooling}")

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.freeze_text_encoder:
            with torch.no_grad():
                out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        pooled = self._pool_query_tokens(token_emb, attention_mask)
        q = self.query_projection(pooled)
        return q

    def encode_text_multi(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if self.freeze_text_encoder:
            with torch.no_grad():
                out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        token_emb = out.last_hidden_state
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

    def compute_frame_contrastive_loss(
        self,
        query_embedding: torch.Tensor,
        frame_embeddings: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
    ):
        q = F.normalize(query_embedding, dim=-1)
        frames = F.normalize(frame_embeddings, dim=-1)
        B, N, _ = frames.shape
        losses = []

        for i in range(B):
            valid_len = int(feature_mask[i].sum().item()) if feature_mask is not None else N
            if valid_len <= 0:
                continue
            s = int(gt_start_idx[i].item())
            e = int(gt_end_idx[i].item())
            s = max(0, min(s, valid_len - 1))
            e = max(s, min(e, valid_len - 1))

            scores_self = frames[i, :valid_len] @ q[i]
            pos_score = scores_self[s : e + 1].max()

            neg_scores = []
            for j in range(B):
                if j == i:
                    continue
                valid_j = int(feature_mask[j].sum().item()) if feature_mask is not None else N
                if valid_j <= 0:
                    continue
                neg_scores.append((frames[j, :valid_j] @ q[i]).max())

            if len(neg_scores) == 0:
                outside_mask = torch.ones(valid_len, dtype=torch.bool, device=frames.device)
                outside_mask[s : e + 1] = False
                if outside_mask.any():
                    outside_scores = scores_self[outside_mask]
                    k = min(16, int(outside_scores.numel()))
                    neg_scores = list(torch.topk(outside_scores, k=k).values)
                else:
                    continue

            neg_scores_tensor = torch.stack(neg_scores)
            logits = torch.cat([pos_score.view(1), neg_scores_tensor], dim=0) / self.temperature
            target = torch.zeros(1, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)

            if self.weak_positive_weight > 0:
                outside_mask = torch.ones(valid_len, dtype=torch.bool, device=frames.device)
                outside_mask[s : e + 1] = False
                if outside_mask.any():
                    weak_pos_score = scores_self[outside_mask].max()
                    logits_w = torch.cat([weak_pos_score.view(1), neg_scores_tensor.detach()], dim=0) / self.temperature
                    loss = loss + (self.weak_positive_weight * F.cross_entropy(logits_w.unsqueeze(0), target))

            losses.append(loss)

        if len(losses) == 0:
            return query_embedding.sum() * 0.0
        return torch.stack(losses).mean()

    def compute_event_contrastive_loss(
        self,
        query_embedding: torch.Tensor,
        event_embeddings: torch.Tensor,
        event_spans_batch: List[List[Tuple[int, int]]],
        event_mask: torch.Tensor,
        frame_embeddings: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
    ):
        q = F.normalize(query_embedding, dim=-1)
        events = F.normalize(event_embeddings, dim=-1)
        frames = F.normalize(frame_embeddings, dim=-1)
        B, M, _ = events.shape
        losses = []

        for i in range(B):
            spans_i = event_spans_batch[i]
            if len(spans_i) == 0:
                continue

            valid_len = frame_embeddings.shape[1]
            s = int(gt_start_idx[i].item())
            e = int(gt_end_idx[i].item())
            s = max(0, min(s, valid_len - 1))
            e = max(s, min(e, valid_len - 1))

            frame_scores_gt = frames[i, s : e + 1] @ q[i]
            pos_frame_idx = s + int(torch.argmax(frame_scores_gt).item())

            pos_event_idx = find_event_containing_frame(spans_i, pos_frame_idx)
            if pos_event_idx is None or pos_event_idx >= M or not bool(event_mask[i, pos_event_idx].item()):
                pos_event_idx, _ = find_best_iou_event(spans_i, s, e)
            if pos_event_idx is None or pos_event_idx >= M or not bool(event_mask[i, pos_event_idx].item()):
                continue

            pos_score = events[i, pos_event_idx] @ q[i]
            neg_scores = []

            for j in range(B):
                if j == i:
                    continue
                valid_j = event_mask[j].bool()
                if valid_j.any():
                    neg_scores.append((events[j, valid_j] @ q[i]).max())

            if len(neg_scores) == 0:
                valid_i = event_mask[i].bool()
                event_scores_i = events[i, valid_i] @ q[i]
                valid_indices = torch.where(valid_i)[0].tolist()
                local_neg_scores = []
                gt_span = (s, e)
                for local_idx, global_event_idx in enumerate(valid_indices):
                    if global_event_idx == pos_event_idx:
                        continue
                    if global_event_idx < len(spans_i):
                        overlap = max(0, min(spans_i[global_event_idx][1], gt_span[1]) - max(spans_i[global_event_idx][0], gt_span[0]) + 1)
                        if overlap <= 0:
                            local_neg_scores.append(event_scores_i[local_idx])
                if len(local_neg_scores) == 0:
                    continue
                local_neg_scores_tensor = torch.stack(local_neg_scores)
                k = min(16, int(local_neg_scores_tensor.numel()))
                neg_scores = list(torch.topk(local_neg_scores_tensor, k=k).values)

            neg_scores_tensor = torch.stack(neg_scores)
            logits = torch.cat([pos_score.view(1), neg_scores_tensor], dim=0) / self.temperature
            target = torch.zeros(1, dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits.unsqueeze(0), target))

        if len(losses) == 0:
            return query_embedding.sum() * 0.0
        return torch.stack(losses).mean()

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
        if self.use_modality_specific_query:
            q_dict = self.encode_text_multi(input_ids, attention_mask)
            q = q_dict["visual"]
        else:
            q = self.encode_text(input_ids, attention_mask)
        h, g, event_mask, all_spans = self.encode_video_batch(features, feature_mask, return_spans=True)

        q_norm = F.normalize(q, dim=-1) if self.normalize_embeddings else q
        h_norm = F.normalize(h, dim=-1) if self.normalize_embeddings else h
        g_norm = F.normalize(g, dim=-1) if self.normalize_embeddings else g

        frame_scores = torch.einsum("bd,bnd->bn", q_norm, h_norm)
        event_scores = torch.einsum("bd,bmd->bm", q_norm, g_norm)
        frame_scores = frame_scores.masked_fill(~feature_mask.bool(), -1e4)
        event_scores = event_scores.masked_fill(~event_mask.bool(), -1e4)

        out = {
            "query_embedding": q,
            "frame_embeddings": h,
            "event_embeddings": g,
            "event_spans": all_spans,
            "event_mask": event_mask,
            "frame_scores": frame_scores,
            "event_scores": event_scores,
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
        if self.use_modality_specific_query:
            q_dict = self.encode_text_multi(input_ids, attention_mask)
            if modality not in q_dict:
                raise ValueError(f"Unknown modality '{modality}'. Available modalities: {list(q_dict.keys())}")
            q = q_dict[modality]
        else:
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
