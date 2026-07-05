
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
        query_pooling: str = "cls",
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
        event_strategy: str = "tsm",
        event_kmeans_num_events: int = 10,
        event_window_size: int = 5,
        tsm_window_size: int = 4,
        tsm_threshold_alpha: float = 0.5,
        min_event_len: int = 3,
        max_event_len: int = 30,
        lambda_event: float = 0.8,
        use_hard_negative: bool = True,
        lambda_hard: float = 1.0,
        use_weak_positive: bool = True,
        lambda_weak: float = 0.1,
        lambda_weak_event: Optional[float] = None,
        weak_positive_margin: int = 10,
        temperature: float = 0.07,
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
            tsm_window_size=tsm_window_size,
            tsm_threshold_alpha=tsm_threshold_alpha,
            min_event_len=min_event_len,
            max_event_len=max_event_len,
            lambda_event=lambda_event,
            use_hard_negative=use_hard_negative,
            lambda_hard=lambda_hard,
            use_weak_positive=use_weak_positive,
            lambda_weak=lambda_weak,
            lambda_weak_event=lambda_weak if lambda_weak_event is None else lambda_weak_event,
            weak_positive_margin=weak_positive_margin,
            temperature=temperature,
        )
        self.d_model = d_model
        self.lambda_event = lambda_event
        self.use_hard_negative = use_hard_negative
        self.lambda_hard = lambda_hard
        self.use_weak_positive = use_weak_positive
        self.lambda_weak = lambda_weak
        self.lambda_weak_event = lambda_weak if lambda_weak_event is None else lambda_weak_event
        self.weak_positive_margin = weak_positive_margin
        self.tsm_window_size = tsm_window_size
        self.tsm_threshold_alpha = tsm_threshold_alpha
        self.min_event_len = min_event_len
        self.max_event_len = max_event_len
        self.max_frames = max_frames
        self.max_events = max_events
        self.event_strategy = event_strategy
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

    def _pool_events_for_one_video(self, h_one: torch.Tensor, valid_len: int):
        h_valid = h_one[:valid_len]
        spans = self.event_reasoner.detect_event_spans(h_valid)
        event_vecs = []
        for s, e in spans:
            event_vecs.append(h_valid[s:e + 1].max(dim=0).values)
        events = torch.stack(event_vecs, dim=0) if event_vecs else h_valid.max(dim=0, keepdim=True).values
        return events, spans

    def encode_video_batch(self, features: torch.Tensor, feature_mask: torch.Tensor, return_spans: bool = True):
        """Encode batch of videos to contextual frame and event embeddings.

        Args:
            features: [B, N, D_raw]
            feature_mask: [B, N] bool, True for valid segments
        Returns:
            h: [B, N, d_model]
            g: [B, M_max, d_model]
            event_mask: [B, M_max]
            all_spans: list[list[(start, end)]]
        """
        B, N, _ = features.shape
        h = self._project_and_encode_frames(features, feature_mask)
        all_event_vecs = []
        all_spans = []
        max_m = 1
        for b in range(B):
            valid_len = int(feature_mask[b].sum().item())
            events, spans = self._pool_events_for_one_video(h[b], valid_len)
            all_event_vecs.append(events)
            all_spans.append(spans)
            max_m = max(max_m, events.shape[0])

        max_m = min(max_m, self.max_events)
        event_tensor = h.new_zeros((B, max_m, self.d_model))
        event_mask = torch.zeros((B, max_m), dtype=torch.bool, device=h.device)
        clipped_spans = []
        for b, events in enumerate(all_event_vecs):
            m = min(events.shape[0], max_m)
            event_tensor[b, :m] = events[:m]
            event_mask[b, :m] = True
            clipped_spans.append(all_spans[b][:m])

        pos = self._valid_positions(max_m, self.max_events, h.device)
        event_tensor = event_tensor + self.event_pos_embed(pos).unsqueeze(0)
        event_tensor = event_tensor * event_mask.unsqueeze(-1).to(event_tensor.dtype)
        g = self.event_encoder(event_tensor, event_mask)
        return h, g, event_mask, clipped_spans

    @staticmethod
    def _contrastive_loss(q: torch.Tensor, pos: torch.Tensor, logit_scale: torch.Tensor):
        qn = F.normalize(q, dim=-1)
        pn = F.normalize(pos, dim=-1)
        scale = logit_scale.exp().clamp(max=100.0)
        scores = scale * (qn @ pn.t())
        labels = torch.arange(q.shape[0], device=q.device)
        return 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.t(), labels)), scores

    @staticmethod
    def _contrastive_loss_inbatch_plus_hard(
        q: torch.Tensor,
        pos: torch.Tensor,
        hard_neg: torch.Tensor,
        logit_scale: torch.Tensor,
        hard_neg_valid_mask: Optional[torch.Tensor] = None,
    ):
        qn = F.normalize(q, dim=-1)
        pn = F.normalize(pos, dim=-1)
        hn = F.normalize(hard_neg, dim=-1)

        scale = logit_scale.exp().clamp(max=100.0)
        inbatch_logits = qn @ pn.t()
        hard_logits = (qn * hn).sum(dim=-1, keepdim=True)
        if hard_neg_valid_mask is not None:
            hard_logits = hard_logits.masked_fill(~hard_neg_valid_mask.view(-1, 1), -1e4)
        logits = scale * torch.cat([inbatch_logits, hard_logits], dim=1)
        labels = torch.arange(q.shape[0], device=q.device)
        loss_q2v = F.cross_entropy(logits, labels)
        return loss_q2v, logits

    @staticmethod
    def _weak_positive_loss(
        q: torch.Tensor,
        weak_pos: torch.Tensor,
        logit_scale: torch.Tensor,
        weak_valid_mask: Optional[torch.Tensor] = None,
    ):
        qn = F.normalize(q, dim=-1)
        wn = F.normalize(weak_pos, dim=-1)
        scale = logit_scale.exp().clamp(max=100.0)
        sim = scale * (qn * wn).sum(dim=-1)
        loss_per_sample = F.softplus(-sim)
        if weak_valid_mask is not None:
            mask = weak_valid_mask.to(loss_per_sample.dtype)
            denom = mask.sum().clamp(min=1.0)
            return (loss_per_sample * mask).sum() / denom
        return loss_per_sample.mean()

    @staticmethod
    def _find_event_containing(spans: List[Tuple[int, int]], idx: int):
        for j, (s, e) in enumerate(spans):
            if s <= idx <= e:
                return j
        # fallback: max temporal overlap with a 1-frame point
        if not spans:
            return 0
        centers = [abs((s + e) / 2.0 - idx) for s, e in spans]
        return int(min(range(len(centers)), key=lambda k: centers[k]))

    def _select_positives(self, q: torch.Tensor, h: torch.Tensor, g: torch.Tensor, gt_start_idx: torch.Tensor, gt_end_idx: torch.Tensor, all_spans):
        B = q.shape[0]
        pos_frames = []
        pos_events = []
        pos_frame_indices = []
        pos_event_indices = []
        qn = F.normalize(q.detach(), dim=-1)
        hn = F.normalize(h.detach(), dim=-1)
        for b in range(B):
            s = int(gt_start_idx[b].item())
            e = int(gt_end_idx[b].item())
            valid_n = h.shape[1]
            s = max(0, min(s, valid_n - 1))
            e = max(s, min(e, valid_n - 1))
            sims = hn[b, s:e + 1] @ qn[b]
            offset = int(torch.argmax(sims).item())
            pos_frame_idx = s + offset
            event_idx = self._find_event_containing(all_spans[b], pos_frame_idx)
            event_idx = min(event_idx, g.shape[1] - 1)
            pos_frames.append(h[b, pos_frame_idx])
            pos_events.append(g[b, event_idx])
            pos_frame_indices.append(pos_frame_idx)
            pos_event_indices.append(event_idx)
        return torch.stack(pos_frames), torch.stack(pos_events), pos_frame_indices, pos_event_indices

    def _mine_hard_negative_frames(
        self,
        q: torch.Tensor,
        h: torch.Tensor,
        feature_mask: torch.Tensor,
        video_ids=None,
    ):
        B, N, _ = h.shape
        if B < 2:
            return h[:, 0], [None] * B, torch.zeros(B, dtype=torch.bool, device=h.device)
        qn = F.normalize(q.detach(), dim=-1)
        hn = F.normalize(h.detach(), dim=-1)

        hard_neg_frames = []
        hard_neg_indices = []
        valid_list = []

        for i in range(B):
            sim = torch.einsum("d,bnd->bn", qn[i], hn)
            sim = sim.masked_fill(~feature_mask.bool(), float("-inf"))
            if video_ids is not None:
                same_video_mask = torch.zeros(B, dtype=torch.bool, device=h.device)
                for j in range(B):
                    if video_ids[j] == video_ids[i]:
                        same_video_mask[j] = True
                sim = sim.masked_fill(same_video_mask.view(B, 1), float("-inf"))
            else:
                sim[i, :] = float("-inf")

            if not torch.isfinite(sim).any():
                hard_neg_frames.append(h[i, 0])
                hard_neg_indices.append(None)
                valid_list.append(False)
                continue

            flat_idx = torch.argmax(sim.reshape(-1))
            neg_b = int(flat_idx // N)
            neg_t = int(flat_idx % N)
            hard_neg_frames.append(h[neg_b, neg_t])
            hard_neg_indices.append((neg_b, neg_t))
            valid_list.append(True)

        hard_neg_valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=h.device)
        return torch.stack(hard_neg_frames), hard_neg_indices, hard_neg_valid_mask

    def _mine_hard_negative_events(
        self,
        q: torch.Tensor,
        g: torch.Tensor,
        event_mask: torch.Tensor,
        video_ids=None,
    ):
        B, M, _ = g.shape
        if B < 2:
            return g[:, 0], [None] * B, torch.zeros(B, dtype=torch.bool, device=g.device)
        qn = F.normalize(q.detach(), dim=-1)
        gn = F.normalize(g.detach(), dim=-1)

        hard_neg_events = []
        hard_neg_indices = []
        valid_list = []

        for i in range(B):
            sim = torch.einsum("d,bmd->bm", qn[i], gn)
            sim = sim.masked_fill(~event_mask.bool(), float("-inf"))
            if video_ids is not None:
                same_video_mask = torch.zeros(B, dtype=torch.bool, device=g.device)
                for j in range(B):
                    if video_ids[j] == video_ids[i]:
                        same_video_mask[j] = True
                sim = sim.masked_fill(same_video_mask.view(B, 1), float("-inf"))
            else:
                sim[i, :] = float("-inf")

            if not torch.isfinite(sim).any():
                hard_neg_events.append(g[i, 0])
                hard_neg_indices.append(None)
                valid_list.append(False)
                continue

            flat_idx = torch.argmax(sim.reshape(-1))
            neg_b = int(flat_idx // M)
            neg_m = int(flat_idx % M)
            hard_neg_events.append(g[neg_b, neg_m])
            hard_neg_indices.append((neg_b, neg_m))
            valid_list.append(True)

        hard_neg_valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=g.device)
        return torch.stack(hard_neg_events), hard_neg_indices, hard_neg_valid_mask

    def _select_weak_positives(
        self,
        q: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        feature_mask: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        all_spans,
    ):
        B, _, _ = h.shape
        weak_frames = []
        weak_events = []
        weak_frame_indices = []
        weak_event_indices = []
        weak_valid = []

        qn = F.normalize(q.detach(), dim=-1)
        hn = F.normalize(h.detach(), dim=-1)

        for b in range(B):
            valid_n = int(feature_mask[b].sum().item())
            s = int(gt_start_idx[b].item())
            e = int(gt_end_idx[b].item())
            s = max(0, min(s, valid_n - 1))
            e = max(s, min(e, valid_n - 1))

            candidate_mask = torch.zeros(valid_n, dtype=torch.bool, device=h.device)
            margin = int(self.weak_positive_margin)
            left_start = max(0, s - margin)
            left_end = s
            right_start = e + 1
            right_end = min(valid_n, e + 1 + margin)

            if left_start < left_end:
                candidate_mask[left_start:left_end] = True
            if right_start < right_end:
                candidate_mask[right_start:right_end] = True

            if candidate_mask.any():
                sims = hn[b, :valid_n] @ qn[b]
                sims = sims.masked_fill(~candidate_mask, float("-inf"))
                weak_idx = int(torch.argmax(sims).item())
                is_valid = True
            else:
                weak_idx = s
                is_valid = False

            weak_event_idx = self._find_event_containing(all_spans[b], weak_idx)
            weak_event_idx = min(weak_event_idx, g.shape[1] - 1)

            weak_frames.append(h[b, weak_idx])
            weak_events.append(g[b, weak_event_idx])
            weak_frame_indices.append(weak_idx if is_valid else None)
            weak_event_indices.append(weak_event_idx if is_valid else None)
            weak_valid.append(is_valid)

        weak_valid_mask = torch.tensor(weak_valid, dtype=torch.bool, device=h.device)

        return (
            torch.stack(weak_frames),
            torch.stack(weak_events),
            weak_frame_indices,
            weak_event_indices,
            weak_valid_mask,
        )

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gt_start_idx: torch.Tensor,
        gt_end_idx: torch.Tensor,
        video_ids=None,
    ):
        if self.use_modality_specific_query:
            q_dict = self.encode_text_multi(input_ids, attention_mask)
            q = q_dict["visual"]
        else:
            q = self.encode_text(input_ids, attention_mask)
        h, g, event_mask, all_spans = self.encode_video_batch(features, feature_mask, return_spans=True)
        pos_frames, pos_events, pos_frame_indices, pos_event_indices = self._select_positives(q, h, g, gt_start_idx, gt_end_idx, all_spans)
        loss_frame_base, scores_frame_base = self._contrastive_loss(q, pos_frames, self.logit_scale_frame)
        loss_event_base, scores_event_base = self._contrastive_loss(q, pos_events, self.logit_scale_event)

        loss_frame_hard = q.new_zeros(())
        loss_event_hard = q.new_zeros(())
        scores_frame_hard = None
        scores_event_hard = None
        hard_neg_frame_indices = [None] * q.shape[0]
        hard_neg_event_indices = [None] * q.shape[0]
        hard_neg_frame_valid_mask = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
        hard_neg_event_valid_mask = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)

        if self.use_hard_negative and q.shape[0] >= 2:
            hard_neg_frames, hard_neg_frame_indices, hard_neg_frame_valid_mask = self._mine_hard_negative_frames(
                q=q, h=h, feature_mask=feature_mask, video_ids=video_ids
            )
            hard_neg_events, hard_neg_event_indices, hard_neg_event_valid_mask = self._mine_hard_negative_events(
                q=q, g=g, event_mask=event_mask, video_ids=video_ids
            )
            loss_frame_hard, scores_frame_hard = self._contrastive_loss_inbatch_plus_hard(
                q=q,
                pos=pos_frames,
                hard_neg=hard_neg_frames,
                logit_scale=self.logit_scale_frame,
                hard_neg_valid_mask=hard_neg_frame_valid_mask,
            )
            loss_event_hard, scores_event_hard = self._contrastive_loss_inbatch_plus_hard(
                q=q,
                pos=pos_events,
                hard_neg=hard_neg_events,
                logit_scale=self.logit_scale_event,
                hard_neg_valid_mask=hard_neg_event_valid_mask,
            )

        loss_frame = loss_frame_base + self.lambda_hard * loss_frame_hard
        loss_event = loss_event_base + self.lambda_hard * loss_event_hard

        if self.use_weak_positive:
            weak_frames, weak_events, weak_frame_indices, weak_event_indices, weak_valid_mask = self._select_weak_positives(
                q=q,
                h=h,
                g=g,
                feature_mask=feature_mask,
                gt_start_idx=gt_start_idx,
                gt_end_idx=gt_end_idx,
                all_spans=all_spans,
            )
            loss_weak_frame = self._weak_positive_loss(q=q, weak_pos=weak_frames, logit_scale=self.logit_scale_frame, weak_valid_mask=weak_valid_mask)
            loss_weak_event = self._weak_positive_loss(q=q, weak_pos=weak_events, logit_scale=self.logit_scale_event, weak_valid_mask=weak_valid_mask)
        else:
            loss_weak_frame = q.new_zeros(())
            loss_weak_event = q.new_zeros(())
            weak_frame_indices = [None] * q.shape[0]
            weak_event_indices = [None] * q.shape[0]
            weak_valid_mask = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
        loss = (
            loss_frame
            + self.lambda_event * loss_event
            + self.lambda_weak * loss_weak_frame
            + self.lambda_weak_event * loss_weak_event
        )
        return {
            "loss": loss,
            "loss_frame": loss_frame.detach(),
            "loss_event": loss_event.detach(),
            "loss_frame_base": loss_frame_base.detach(),
            "loss_event_base": loss_event_base.detach(),
            "loss_frame_hard": loss_frame_hard.detach(),
            "loss_event_hard": loss_event_hard.detach(),
            "loss_weak_frame": loss_weak_frame.detach(),
            "loss_weak_event": loss_weak_event.detach(),
            "scores_frame": scores_frame_base.detach(),
            "scores_event": scores_event_base.detach(),
            "scores_frame_hard": scores_frame_hard.detach() if scores_frame_hard is not None else None,
            "scores_event_hard": scores_event_hard.detach() if scores_event_hard is not None else None,
            "pos_frame_indices": pos_frame_indices,
            "pos_event_indices": pos_event_indices,
            "hard_neg_frame_indices": hard_neg_frame_indices,
            "hard_neg_event_indices": hard_neg_event_indices,
            "hard_neg_frame_valid_mask": hard_neg_frame_valid_mask.detach(),
            "hard_neg_event_valid_mask": hard_neg_event_valid_mask.detach(),
            "weak_frame_indices": weak_frame_indices,
            "weak_event_indices": weak_event_indices,
            "weak_valid_mask": weak_valid_mask.detach(),
            "event_spans": all_spans,
        }

    @torch.inference_mode()
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True, modality: str = "visual"):
        if self.use_modality_specific_query:
            q_dict = self.encode_text_multi(input_ids, attention_mask)
            if modality not in q_dict:
                raise ValueError(f"Unknown modality '{modality}'. Available modalities: {list(q_dict.keys())}")
            q = q_dict[modality]
        else:
            q = self.encode_text(input_ids, attention_mask)
        if normalize:
            q = F.normalize(q, dim=-1)
        return q

    @torch.inference_mode()
    def encode_video(self, features: torch.Tensor, feature_mask: torch.Tensor, normalize: bool = True):
        return self.encode_video_trainable(features=features, feature_mask=feature_mask, normalize=normalize)

    def encode_video_trainable(self, features: torch.Tensor, feature_mask: torch.Tensor, normalize: bool = False):
        h, g, event_mask, all_spans = self.encode_video_batch(features, feature_mask, return_spans=True)
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
