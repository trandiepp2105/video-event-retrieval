from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def build_window_events(valid_len: int, window_size: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    if valid_len <= 0:
        return []
    if window_size <= 0:
        raise ValueError(f"event_window_size must be > 0, got {window_size}")

    if stride is None:
        stride = window_size
    stride = max(1, int(stride))

    if valid_len <= window_size:
        return [(0, valid_len - 1)]

    spans: List[Tuple[int, int]] = []
    for start in range(0, valid_len, stride):
        end = min(start + window_size - 1, valid_len - 1)
        if start <= end:
            spans.append((start, end))
        if end == valid_len - 1:
            break
    return spans


def build_multiscale_window_events(
    valid_len: int,
    window_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64),
    stride_ratio: float = 0.5,
    max_events: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if valid_len <= 0:
        return []

    spans: List[Tuple[int, int]] = []
    for window_size in window_sizes:
        window_size = int(window_size)
        if window_size <= 0:
            continue
        if valid_len <= window_size:
            spans.append((0, valid_len - 1))
            continue

        stride = max(1, int(window_size * stride_ratio))
        spans.extend(build_window_events(valid_len=valid_len, window_size=window_size, stride=stride))

    seen = set()
    unique_spans: List[Tuple[int, int]] = []
    for span in spans:
        if span not in seen:
            seen.add(span)
            unique_spans.append(span)
    spans = unique_spans

    if max_events is not None and len(spans) > max_events:
        if max_events <= 0:
            return []
        if max_events == 1:
            return [spans[0]]
        indices = []
        for idx in range(max_events):
            position = round(idx * (len(spans) - 1) / (max_events - 1))
            indices.append(int(position))
        spans = [spans[idx] for idx in indices]

    if len(spans) == 0:
        spans = [(0, valid_len - 1)]
    return spans


class EventReasoner:
    def __init__(
        self,
        strategy: str = "window",
        tsm_window_size: int = 4,
        tsm_threshold_alpha: float = 0.5,
        min_event_len: int = 3,
        max_event_len: int = 30,
        kmeans_num_events: int = 10,
        window_size: int = 8,
        stride: Optional[int] = None,
        window_sizes: Tuple[int, ...] = (4, 8, 16, 32, 64),
        stride_ratio: float = 0.5,
        max_events: int = 1024,
    ):
        self.strategy = strategy
        self.tsm_window_size = tsm_window_size
        self.tsm_threshold_alpha = tsm_threshold_alpha
        self.min_event_len = min_event_len
        self.max_event_len = max_event_len
        self.kmeans_num_events = kmeans_num_events
        self.window_size = window_size
        self.stride = stride
        self.window_sizes = tuple(window_sizes)
        self.stride_ratio = stride_ratio
        self.max_events = max_events

    @staticmethod
    def _mean_square_block(tsm: torch.Tensor, start: int, end: int, exclude_diag: bool = True):
        block = tsm[start:end, start:end]
        if block.numel() == 0:
            return torch.tensor(0.0, device=tsm.device)
        if exclude_diag and block.shape[0] > 1:
            total = block.sum() - torch.diagonal(block).sum()
            denom = block.numel() - block.shape[0]
            return total / max(denom, 1)
        return block.mean()

    def _boundaries_to_spans(self, boundaries: List[int], n: int):
        boundaries = sorted({int(b) for b in boundaries if 0 < int(b) < n})
        starts = [0] + boundaries
        ends = boundaries + [n]
        spans = []
        for s, e in zip(starts, ends):
            if e <= s:
                continue
            spans.append((int(s), int(e - 1)))
        return spans

    def _split_too_long_spans(self, spans: List[Tuple[int, int]]):
        max_len = max(1, int(self.max_event_len))
        out = []
        for s, e in spans:
            cur = int(s)
            while cur <= e:
                end = min(e, cur + max_len - 1)
                out.append((cur, end))
                cur = end + 1
        return out

    def _merge_too_short_spans(self, spans: List[Tuple[int, int]], n: int):
        if not spans:
            return [(0, n - 1)] if n > 0 else []

        min_len = max(1, int(self.min_event_len))
        merged = [list(spans[0])]
        for s, e in spans[1:]:
            cur_len = merged[-1][1] - merged[-1][0] + 1
            new_len = e - s + 1
            if cur_len < min_len:
                merged[-1][1] = e
            elif new_len < min_len:
                merged[-1][1] = e
            else:
                merged.append([s, e])

        if len(merged) > 1:
            last_len = merged[-1][1] - merged[-1][0] + 1
            if last_len < min_len:
                merged[-2][1] = merged[-1][1]
                merged.pop()

        return [(int(s), int(e)) for s, e in merged]

    def _sanitize_spans(self, spans: List[Tuple[int, int]], n: int):
        if n <= 0:
            return []
        clean = []
        for s, e in spans:
            s = max(0, min(int(s), n - 1))
            e = max(s, min(int(e), n - 1))
            clean.append((s, e))
        clean.sort()
        if not clean:
            return [(0, n - 1)]

        merged = [list(clean[0])]
        for s, e in clean[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(int(s), int(e)) for s, e in merged]

    def _finalize_spans(self, spans: List[Tuple[int, int]], n: int):
        spans = self._sanitize_spans(spans, n)
        spans = self._merge_too_short_spans(spans, n)
        spans = self._split_too_long_spans(spans)
        spans = self._sanitize_spans(spans, n)
        return spans if spans else [(0, n - 1)]

    def _compute_contrastive_boundary_scores(self, h_valid: torch.Tensor):
        n = int(h_valid.shape[0])
        if n <= 0:
            return [], None

        z = F.normalize(h_valid.detach().float(), dim=-1)
        tsm = z @ z.t()
        w = max(1, int(self.tsm_window_size))
        candidate_ts: List[int] = []
        candidate_scores = []

        for t in range(self.min_event_len, n - self.min_event_len + 1):
            l0, l1 = max(0, t - w), t
            r0, r1 = t, min(n, t + w)
            if l1 <= l0 or r1 <= r0:
                continue

            ll = self._mean_square_block(tsm, l0, l1, exclude_diag=True)
            rr = self._mean_square_block(tsm, r0, r1, exclude_diag=True)
            lr = tsm[l0:l1, r0:r1].mean()
            rl = tsm[r0:r1, l0:l1].mean()
            candidate_ts.append(t)
            candidate_scores.append(ll + rr - lr - rl)

        if not candidate_ts:
            return [], None
        return candidate_ts, torch.stack(candidate_scores)

    def _build_contrastive_convolution_kernel(self, device: torch.device, dtype: torch.dtype):
        w = max(1, int(self.tsm_window_size))
        kernel_size = (2 * w) + 1
        kernel = torch.zeros(kernel_size, kernel_size, dtype=dtype, device=device)

        pos_mask = torch.zeros_like(kernel, dtype=torch.bool)
        neg_mask = torch.zeros_like(kernel, dtype=torch.bool)

        pos_mask[:w, :w] = True
        pos_mask[w + 1 :, w + 1 :] = True
        neg_mask[:w, w + 1 :] = True
        neg_mask[w + 1 :, :w] = True

        diag = torch.eye(kernel_size, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~diag

        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())
        if pos_count > 0:
            kernel[pos_mask] = 1.0 / float(pos_count)
        if neg_count > 0:
            kernel[neg_mask] = -1.0 / float(neg_count)
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _compute_contrastive_convolution_scores(self, h_valid: torch.Tensor):
        n = int(h_valid.shape[0])
        if n <= 0:
            return [], None

        z = F.normalize(h_valid.detach().float(), dim=-1)
        tsm = z @ z.t()
        kernel = self._build_contrastive_convolution_kernel(device=tsm.device, dtype=tsm.dtype)
        padding = kernel.shape[-1] // 2
        score_map = F.conv2d(
            tsm.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=padding,
        ).squeeze(0).squeeze(0)

        candidate_ts: List[int] = []
        candidate_scores = []
        for t in range(self.min_event_len, n - self.min_event_len):
            candidate_ts.append(t)
            candidate_scores.append(score_map[t, t])

        if not candidate_ts:
            return [], None
        return candidate_ts, torch.stack(candidate_scores)

    def detect_window_spans(self, h_valid: torch.Tensor):
        n = int(h_valid.shape[0])
        return build_window_events(valid_len=n, window_size=int(self.window_size), stride=self.stride)

    def detect_multiscale_window_spans(self, h_valid: torch.Tensor):
        n = int(h_valid.shape[0])
        return build_multiscale_window_events(
            valid_len=n,
            window_sizes=self.window_sizes,
            stride_ratio=self.stride_ratio,
            max_events=self.max_events,
        )

    def detect_tsm_spans(self, h_valid: torch.Tensor) -> List[Tuple[int, int]]:
        with torch.no_grad():
            n = int(h_valid.shape[0])
            if n <= 0:
                return []
            if n <= self.min_event_len * 2:
                return [(0, n - 1)]

            candidate_ts, scores = self._compute_contrastive_boundary_scores(h_valid)

            if not candidate_ts:
                boundaries = []
            else:
                threshold = scores.mean() + self.tsm_threshold_alpha * scores.std(unbiased=False)
                boundaries_with_scores = []
                for i, t in enumerate(candidate_ts):
                    s = scores[i]
                    left_ok = (i == 0) or (s >= scores[i - 1])
                    right_ok = (i == len(candidate_ts) - 1) or (s >= scores[i + 1])
                    if s > threshold and left_ok and right_ok:
                        boundaries_with_scores.append((t, float(s.item())))

                boundaries_with_scores.sort(key=lambda x: x[1], reverse=True)
                selected = []
                for t, _ in boundaries_with_scores:
                    if all(abs(t - b) >= self.min_event_len for b in selected):
                        selected.append(t)
                boundaries = sorted(selected)

            filtered = []
            prev = 0
            for b in boundaries:
                if b - prev >= self.min_event_len and n - b >= self.min_event_len:
                    filtered.append(b)
                    prev = b
            boundaries = filtered

            final_boundaries = []
            prev = 0
            for b in boundaries + [n]:
                while b - prev > self.max_event_len:
                    prev = prev + self.max_event_len
                    final_boundaries.append(prev)
                if b < n:
                    final_boundaries.append(b)
                    prev = b

            return self._boundaries_to_spans(final_boundaries, n) or [(0, n - 1)]

    def detect_contrastive_convolution_spans(self, h_valid: torch.Tensor):
        with torch.no_grad():
            n = int(h_valid.shape[0])
            if n <= 0:
                return []
            if n <= self.min_event_len * 2:
                return [(0, n - 1)]

            candidate_ts, scores = self._compute_contrastive_convolution_scores(h_valid)
            if not candidate_ts or scores is None:
                return [(0, n - 1)]

            threshold = scores.mean() + self.tsm_threshold_alpha * scores.std(unbiased=False)
            boundaries = []
            for i, t in enumerate(candidate_ts):
                s = scores[i]
                left_ok = (i == 0) or (s >= scores[i - 1])
                right_ok = (i == len(candidate_ts) - 1) or (s >= scores[i + 1])
                if s > threshold and left_ok and right_ok:
                    boundaries.append(t)

            spans = self._boundaries_to_spans(boundaries, n)
            return self._finalize_spans(spans, n)

    def detect_kmeans_spans(self, h_valid: torch.Tensor):
        n = int(h_valid.shape[0])
        if n <= 0:
            return []
        if n <= self.min_event_len * 2:
            return [(0, n - 1)]

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return self.detect_window_spans(h_valid)

        with torch.no_grad():
            z = F.normalize(h_valid.detach().float(), dim=-1)
            tsm = z @ z.t()

        pos = torch.linspace(0.0, 1.0, n, device=tsm.device).unsqueeze(1)
        feat = torch.cat([tsm, pos], dim=1).cpu().numpy()
        k = min(self.kmeans_num_events, max(1, n // max(1, self.min_event_len)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(feat)

        spans = []
        start = 0
        for i in range(1, n):
            if labels[i] != labels[i - 1]:
                spans.append((start, i - 1))
                start = i
        spans.append((start, n - 1))

        return self._finalize_spans(spans, n)

    def detect_event_spans(self, h_valid: torch.Tensor):
        if self.strategy == "window":
            return self.detect_window_spans(h_valid)
        if self.strategy == "multiscale_window":
            return self.detect_multiscale_window_spans(h_valid)
        if self.strategy == "tsm":
            spans = self.detect_tsm_spans(h_valid)
        elif self.strategy == "contrastive_convolution":
            spans = self.detect_contrastive_convolution_spans(h_valid)
        elif self.strategy == "kmeans":
            spans = self.detect_kmeans_spans(h_valid)
        else:
            raise ValueError(f"Unsupported event reasoning strategy: {self.strategy}")

        valid_len = int(h_valid.shape[0])
        if valid_len > max(1, int(self.window_size)) * 2 and len(spans) <= 1:
            return build_window_events(valid_len=valid_len, window_size=int(self.window_size), stride=self.stride)
        if len(spans) == 0 and valid_len > 0:
            return [(0, valid_len - 1)]
        return spans
