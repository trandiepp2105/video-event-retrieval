from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch




def _filter_dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


@dataclass
class TrainConfig:
    train_stage: str = "retriever"
    train_manifest: str = "manifests/train_v1.jsonl"
    val_manifest: str = "manifests/val_v1.jsonl"
    feature_dir: str = "visual_features"
    output_dir: str = "checkpoints"
    retriever_checkpoint: Optional[str] = None
    shared_norm_negatives: Optional[str] = None

    text_model_name: str = "roberta-base"
    freeze_text_encoder: bool = True
    query_pooling: str = "cls"
    use_modality_specific_query: bool = False
    modalities: Tuple[str, ...] = ("visual",)

    d_model: int = 768
    frame_layers: int = 2
    event_layers: int = 2
    num_heads: int = 4
    frame_anchor_sizes: Tuple[Any, ...] = (3, 6, 9, "all")
    event_anchor_sizes: Tuple[Any, ...] = (1, 2, 3, "all")
    ff_dim: int = 3072
    dropout: float = 0.1

    max_frames: int = 2048
    max_events: int = 512
    event_strategy: str = "tsm"
    event_kmeans_num_events: int = 10
    event_window_size: int = 5
    tsm_window_size: int = 4
    tsm_threshold_alpha: float = 0.5
    min_event_len: int = 3
    max_event_len: int = 30
    lambda_event: float = 0.8
    use_hard_negative: bool = True
    lambda_hard: float = 1.0
    use_weak_positive: bool = True
    lambda_weak: float = 0.1
    lambda_weak_event: Optional[float] = None
    weak_positive_margin: int = 10
    temperature: float = 0.07
    use_moment_localizer: bool = False
    use_cross_attention: bool = False
    use_event_auxiliary_loss: bool = False
    lambda_event_localizer: float = 0.8
    max_localizer_span_len: int = 64
    use_shared_norm: bool = False
    shared_norm_num_negatives: int = 5
    segment_duration_sec: float = 1.5

    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    amp: bool = True
    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    tokenizer_max_length: int = 64
    seed: int = 42
    save_best_only: bool = False
    best_metric: str = "event_r1_iou_0_5"
    best_metric_mode: str = "max"

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(**_filter_dataclass_kwargs(cls, data))
        cfg.modalities = tuple(cfg.modalities)
        cfg.frame_anchor_sizes = tuple(cfg.frame_anchor_sizes)
        cfg.event_anchor_sizes = tuple(cfg.event_anchor_sizes)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
