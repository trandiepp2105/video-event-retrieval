from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from eventformer_v1_dynamic_tsm.config import TrainConfig, resolve_text_model_source
from eventformer_v1_dynamic_tsm.io_utils import JsonlReader
from eventformer_v1_dynamic_tsm.localizer import EventAwareMomentLocalizer
from eventformer_v1_dynamic_tsm.model import EventFormerV1DynamicTSM


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage retrieval and localization.")
    parser.add_argument("--retriever-config", type=str, default="configs/train_config.json")
    parser.add_argument("--localizer-config", type=str, default="configs/train_config.json")
    parser.add_argument("--retriever-checkpoint", type=str, required=True)
    parser.add_argument("--localizer-checkpoint", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k-videos", type=int, default=10)
    parser.add_argument("--top-k-spans-per-video", type=int, default=5)
    parser.add_argument("--manifest", type=str, default=None)
    return parser.parse_args()


def resolve_feature_path(feature_dir: str, feature_path: str) -> Path:
    path = Path(feature_path)
    if not path.is_absolute():
        path = Path(feature_dir) / path
    return path


def main():
    args = parse_args()
    retr_cfg = TrainConfig.from_json(args.retriever_config)
    loc_cfg = TrainConfig.from_json(args.localizer_config)
    manifest = args.manifest or retr_cfg.val_manifest or retr_cfg.train_manifest
    rows = JsonlReader.read(manifest)
    tokenizer_source, local_only = resolve_text_model_source(retr_cfg.text_model_name, retr_cfg.text_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)

    sample_feature = np.load(resolve_feature_path(retr_cfg.feature_dir, rows[0]["feature_path"]), allow_pickle=True)
    d_raw = int(sample_feature["features"].shape[1])

    retriever = EventFormerV1DynamicTSM(
        d_raw=d_raw,
        d_model=retr_cfg.d_model,
        text_model_name=retr_cfg.text_model_name,
        text_model_path=retr_cfg.text_model_path,
        freeze_text_encoder=retr_cfg.freeze_text_encoder,
        query_pooling=retr_cfg.query_pooling,
        use_modality_specific_query=retr_cfg.use_modality_specific_query,
        modalities=tuple(retr_cfg.modalities),
        max_frames=retr_cfg.max_frames,
        max_events=retr_cfg.max_events,
        frame_layers=retr_cfg.frame_layers,
        event_layers=retr_cfg.event_layers,
        num_heads=retr_cfg.num_heads,
        frame_anchor_sizes=list(retr_cfg.frame_anchor_sizes),
        event_anchor_sizes=list(retr_cfg.event_anchor_sizes),
        ff_dim=retr_cfg.ff_dim,
        dropout=retr_cfg.dropout,
        event_strategy=retr_cfg.event_strategy,
        event_kmeans_num_events=retr_cfg.event_kmeans_num_events,
        event_window_size=retr_cfg.event_window_size,
        tsm_window_size=retr_cfg.tsm_window_size,
        tsm_threshold_alpha=retr_cfg.tsm_threshold_alpha,
        min_event_len=retr_cfg.min_event_len,
        max_event_len=retr_cfg.max_event_len,
    )
    retr_state = torch.load(args.retriever_checkpoint, map_location="cpu")
    retriever.load_state_dict(retr_state.get("state_dict", retr_state.get("model_state_dict", retr_state)), strict=False)
    retriever.eval()

    localizer = EventAwareMomentLocalizer(
        d_raw=d_raw,
        d_model=loc_cfg.d_model,
        text_model_name=loc_cfg.text_model_name,
        text_model_path=loc_cfg.text_model_path,
        freeze_text_encoder=loc_cfg.freeze_text_encoder,
        max_frames=loc_cfg.max_frames,
        max_events=loc_cfg.max_events,
        frame_layers=loc_cfg.frame_layers,
        event_layers=loc_cfg.event_layers,
        num_heads=loc_cfg.num_heads,
        frame_anchor_sizes=tuple(loc_cfg.frame_anchor_sizes),
        event_anchor_sizes=tuple(loc_cfg.event_anchor_sizes),
        ff_dim=loc_cfg.ff_dim,
        dropout=loc_cfg.dropout,
        event_strategy=loc_cfg.event_strategy,
        event_kmeans_num_events=loc_cfg.event_kmeans_num_events,
        event_window_size=loc_cfg.event_window_size,
        tsm_window_size=loc_cfg.tsm_window_size,
        tsm_threshold_alpha=loc_cfg.tsm_threshold_alpha,
        min_event_len=loc_cfg.min_event_len,
        max_event_len=loc_cfg.max_event_len,
        use_cross_attention=loc_cfg.use_cross_attention,
        use_event_auxiliary_loss=loc_cfg.use_event_auxiliary_loss,
        lambda_event_localizer=loc_cfg.lambda_event_localizer,
    )
    loc_state = torch.load(args.localizer_checkpoint, map_location="cpu")
    localizer.load_state_dict(loc_state.get("state_dict", loc_state.get("model_state_dict", loc_state)), strict=False)
    localizer.eval()

    video_rows = {}
    for row in rows:
        video_rows[row["video_id"]] = row
    video_ids = list(video_rows.keys())
    video_embeddings = []
    for video_id in video_ids:
        feat = np.load(resolve_feature_path(retr_cfg.feature_dir, video_rows[video_id]["feature_path"]), allow_pickle=True)["features"].astype("float32")
        feat = torch.from_numpy(feat[: retr_cfg.max_frames]).unsqueeze(0)
        mask = torch.ones(1, feat.shape[1], dtype=torch.bool)
        out = retriever.encode_video(feat, mask, normalize=True)
        valid = out["event_mask"][0]
        emb = out["event_embeddings"][0][valid].mean(dim=0)
        video_embeddings.append(F.normalize(emb, dim=-1))
    video_embeddings = torch.stack(video_embeddings)

    toks = tokenizer([args.query], padding=True, truncation=True, max_length=retr_cfg.tokenizer_max_length, return_tensors="pt")
    q = retriever.encode_query(toks["input_ids"], toks["attention_mask"], normalize=True)[0]
    retrieval_scores = video_embeddings @ q
    topk = min(args.top_k_videos, retrieval_scores.shape[0])
    top_video_indices = torch.topk(retrieval_scores, k=topk).indices.tolist()

    results = []
    for rank, idx in enumerate(top_video_indices, start=1):
        video_id = video_ids[idx]
        feat = np.load(resolve_feature_path(loc_cfg.feature_dir, video_rows[video_id]["feature_path"]), allow_pickle=True)["features"].astype("float32")
        feat = torch.from_numpy(feat[: loc_cfg.max_frames]).unsqueeze(0)
        mask = torch.ones(1, feat.shape[1], dtype=torch.bool)
        out = localizer(
            features=feat,
            feature_mask=mask,
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            gt_start_idx=None,
            gt_end_idx=None,
        )
        spans = localizer.decode_top_spans(
            out["start_logits"],
            out["end_logits"],
            mask,
            topk=args.top_k_spans_per_video,
            max_span_len=loc_cfg.max_localizer_span_len,
        )[0]
        for span in spans:
            final_score = float(retrieval_scores[idx].item()) + float(span["score"])
            results.append({
                "rank": rank,
                "video_id": video_id,
                "start_idx": int(span["start_idx"]),
                "end_idx": int(span["end_idx"]),
                "start_time_sec": float(span["start_idx"] * loc_cfg.segment_duration_sec),
                "end_time_sec": float((span["end_idx"] + 1) * loc_cfg.segment_duration_sec),
                "retrieval_score": float(retrieval_scores[idx].item()),
                "localization_score": float(span["score"]),
                "final_score": final_score,
            })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    print({"query": args.query, "results": results})


if __name__ == "__main__":
    main()
