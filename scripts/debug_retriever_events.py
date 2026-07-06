from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventformer_v1_dynamic_tsm.config import TrainConfig, resolve_text_model_source
from eventformer_v1_dynamic_tsm.model import EventFormerV1DynamicTSM


def load_features(path: str) -> np.ndarray:
    feature_path = Path(path)
    if feature_path.suffix == ".npz":
        data = np.load(feature_path, allow_pickle=True)
        if "features" in data.files:
            x = data["features"]
        else:
            x = data[data.files[0]]
    elif feature_path.suffix == ".npy":
        x = np.load(feature_path, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported feature file: {feature_path}")

    if x.ndim != 2:
        raise ValueError(f"Expected [N, D], got {tuple(x.shape)}")
    return x.astype("float32")


def build_model(cfg: TrainConfig, d_raw: int) -> EventFormerV1DynamicTSM:
    return EventFormerV1DynamicTSM(
        d_raw=d_raw,
        d_model=cfg.d_model,
        text_model_name=cfg.text_model_name,
        text_model_path=cfg.text_model_path,
        freeze_text_encoder=cfg.freeze_text_encoder,
        query_pooling=cfg.query_pooling,
        use_modality_specific_query=cfg.use_modality_specific_query,
        modalities=tuple(cfg.modalities),
        max_frames=cfg.max_frames,
        max_events=cfg.max_events,
        frame_layers=cfg.frame_layers,
        event_layers=cfg.event_layers,
        num_heads=cfg.num_heads,
        frame_anchor_sizes=list(cfg.frame_anchor_sizes),
        event_anchor_sizes=list(cfg.event_anchor_sizes),
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        event_strategy=cfg.event_strategy,
        event_kmeans_num_events=cfg.event_kmeans_num_events,
        event_window_size=cfg.event_window_size,
        event_stride=cfg.event_stride,
        event_window_sizes=tuple(cfg.event_window_sizes),
        event_stride_ratio=cfg.event_stride_ratio,
        event_pooling=cfg.event_pooling,
        tsm_window_size=cfg.tsm_window_size,
        tsm_threshold_alpha=cfg.tsm_threshold_alpha,
        min_event_len=cfg.min_event_len,
        max_event_len=cfg.max_event_len,
        normalize_embeddings=cfg.normalize_embeddings,
        lambda_frame=cfg.lambda_frame,
        lambda_event=cfg.lambda_event,
        weak_positive_weight=cfg.weak_positive_weight,
        temperature=cfg.temperature,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--segment-duration-sec", type=float, default=1.5)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig.from_json(args.config)
    features_np = load_features(args.feature_path)
    print("features shape:", tuple(features_np.shape))

    model = build_model(cfg, d_raw=int(features_np.shape[1]))
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("missing keys:", len(missing))
        print("unexpected keys:", len(unexpected))

    model = model.to(device).eval()
    tokenizer_source, local_only = resolve_text_model_source(cfg.text_model_name, cfg.text_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
    query_inputs = tokenizer(
        args.query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.tokenizer_max_length,
    )
    query_inputs = {key: value.to(device) for key, value in query_inputs.items()}

    features = torch.from_numpy(features_np).unsqueeze(0).to(device)
    feature_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)

    with torch.no_grad():
        q = model.encode_query(**query_inputs)
        video_out = model.encode_video(features=features, feature_mask=feature_mask)

    frames = video_out["frame_embeddings"][0].detach().cpu()
    event_mask = video_out["event_mask"][0].detach().cpu().bool()
    events = video_out["event_embeddings"][0].detach().cpu()[event_mask]
    event_spans = video_out["event_spans"][0]
    q = q[0].detach().cpu()

    print("query_embedding shape:", tuple(q.shape))
    print("frame_embeddings shape:", tuple(frames.shape))
    print("event_embeddings shape:", tuple(events.shape))
    print("num events:", len(event_spans))
    print("first 20 event spans:", event_spans[:20])

    qn = F.normalize(q, dim=-1)
    fn = F.normalize(frames, dim=-1)
    en = F.normalize(events, dim=-1) if events.numel() > 0 else events
    frame_scores = fn @ qn
    event_scores = en @ qn if events.numel() > 0 else torch.empty(0)

    print("\nTop frames:")
    frame_k = min(args.top_k, int(frame_scores.numel()))
    frame_vals, frame_idxs = torch.topk(frame_scores, k=frame_k)
    for rank, (score, idx) in enumerate(zip(frame_vals.tolist(), frame_idxs.tolist()), start=1):
        st = idx * args.segment_duration_sec
        ed = (idx + 1) * args.segment_duration_sec
        print(f"#{rank}: frame={idx} time=[{st:.2f},{ed:.2f}] score={score:.4f}")

    print("\nTop events:")
    if event_scores.numel() == 0:
        print("No events generated.")
    else:
        event_k = min(args.top_k, int(event_scores.numel()))
        event_vals, event_idxs = torch.topk(event_scores, k=event_k)
        for rank, (score, idx) in enumerate(zip(event_vals.tolist(), event_idxs.tolist()), start=1):
            s, e = event_spans[idx]
            st = s * args.segment_duration_sec
            ed = (e + 1) * args.segment_duration_sec
            print(f"#{rank}: event={idx} span=[{s},{e}] time=[{st:.2f},{ed:.2f}] score={score:.4f}")


if __name__ == "__main__":
    main()
