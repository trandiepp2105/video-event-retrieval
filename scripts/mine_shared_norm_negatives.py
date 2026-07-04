from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from eventformer_v1_dynamic_tsm.config import TrainConfig
from eventformer_v1_dynamic_tsm.io_utils import JsonlReader, write_jsonl
from eventformer_v1_dynamic_tsm.model import EventFormerV1DynamicTSM


def parse_args():
    parser = argparse.ArgumentParser(description="Mine negatives for Shared-Norm localizer training.")
    parser.add_argument("--config", type=str, default="configs/train_config.json")
    parser.add_argument("--retriever-checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def resolve_feature_path(feature_dir: str, feature_path: str) -> Path:
    path = Path(feature_path)
    if not path.is_absolute():
        path = Path(feature_dir) / path
    return path


def main():
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    rows = JsonlReader.read(cfg.train_manifest, limit=args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    sample_feature = np.load(resolve_feature_path(cfg.feature_dir, rows[0]["feature_path"]), allow_pickle=True)
    d_raw = int(sample_feature["features"].shape[1])
    model = EventFormerV1DynamicTSM(
        d_raw=d_raw,
        d_model=cfg.d_model,
        text_model_name=cfg.text_model_name,
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
        tsm_window_size=cfg.tsm_window_size,
        tsm_threshold_alpha=cfg.tsm_threshold_alpha,
        min_event_len=cfg.min_event_len,
        max_event_len=cfg.max_event_len,
    )
    ckpt = torch.load(args.retriever_checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    video_rows = {}
    for row in rows:
        video_rows[row["video_id"]] = row

    video_ids = list(video_rows.keys())
    video_embeddings = []
    for video_id in video_ids:
        feat = np.load(resolve_feature_path(cfg.feature_dir, video_rows[video_id]["feature_path"]), allow_pickle=True)["features"].astype("float32")
        feat = torch.from_numpy(feat[: cfg.max_frames]).unsqueeze(0)
        mask = torch.ones(1, feat.shape[1], dtype=torch.bool)
        out = model.encode_video(feat, mask, normalize=True)
        valid = out["event_mask"][0]
        emb = out["event_embeddings"][0][valid].mean(dim=0)
        video_embeddings.append(F.normalize(emb, dim=-1))
    video_embeddings = torch.stack(video_embeddings)

    results = []
    for row in rows:
        toks = tokenizer(
            [row["query"]],
            padding=True,
            truncation=True,
            max_length=cfg.tokenizer_max_length,
            return_tensors="pt",
        )
        q = model.encode_query(toks["input_ids"], toks["attention_mask"], normalize=True)[0]
        scores = video_embeddings @ q
        topk = min(args.pool_size, scores.shape[0])
        ranked = torch.topk(scores, k=topk).indices.tolist()
        negatives = []
        for idx in ranked:
            cand_video_id = video_ids[idx]
            if cand_video_id == row["video_id"]:
                continue
            negatives.append(cand_video_id)
            if len(negatives) >= args.num_negatives:
                break
        results.append({
            "sample_id": row["sample_id"],
            "query": row["query"],
            "positive_video_id": row["video_id"],
            "negative_video_ids": negatives,
        })

    write_jsonl(results, Path(args.output))


if __name__ == "__main__":
    main()
