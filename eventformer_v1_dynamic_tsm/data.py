from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_utils import JsonlReader


class FeatureManifestDataset(Dataset):
    def __init__(self, manifest_path: str, feature_dir: str, limit: Optional[int] = None):
        self.rows = JsonlReader.read(manifest_path, limit)
        self.feature_dir = Path(feature_dir)

    def _resolve_feature_path(self, feature_path: str) -> Path:
        path = Path(feature_path)
        if not path.is_absolute():
            path = self.feature_dir / path
        return path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        feature_path = self._resolve_feature_path(row["feature_path"])
        data = np.load(feature_path, allow_pickle=True)
        features = data["features"].astype("float32")
        return {
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "query": row["query"],
            "features": torch.from_numpy(features),
            "gt_start_idx": int(row["gt_start_idx"]),
            "gt_end_idx": int(row["gt_end_idx"]),
        }


class BatchCollator:
    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_n = max(x["features"].shape[0] for x in items)
        d_raw = items[0]["features"].shape[1]
        batch_size = len(items)
        features = torch.zeros(batch_size, max_n, d_raw, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
        queries: List[str] = []
        sample_ids: List[str] = []
        video_ids: List[str] = []
        gt_s: List[int] = []
        gt_e: List[int] = []

        for b, item in enumerate(items):
            n = item["features"].shape[0]
            features[b, :n] = item["features"]
            mask[b, :n] = True
            queries.append(item["query"])
            sample_ids.append(item["sample_id"])
            video_ids.append(item["video_id"])
            gt_s.append(min(item["gt_start_idx"], n - 1))
            gt_e.append(min(item["gt_end_idx"], n - 1))

        toks = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "sample_ids": sample_ids,
            "video_ids": video_ids,
            "queries": queries,
            "features": features,
            "feature_mask": mask,
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "gt_start_idx": torch.tensor(gt_s, dtype=torch.long),
            "gt_end_idx": torch.tensor(gt_e, dtype=torch.long),
        }


class LocalizerSharedNormDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        feature_dir: str,
        negatives_path: str,
        shared_norm_num_negatives: int = 5,
        max_frames: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        all_rows = JsonlReader.read(manifest_path, limit=None)
        self.rows = all_rows[:limit] if limit is not None else all_rows
        self.feature_dir = Path(feature_dir)
        self.max_frames = max_frames
        self.shared_norm_num_negatives = shared_norm_num_negatives
        negative_rows = JsonlReader.read(negatives_path)
        self.negatives_by_sample_id = {row["sample_id"]: row for row in negative_rows}

        self.video_to_feature_path = {}
        for row in all_rows:
            video_id = row.get("video_id")
            feature_path = row.get("feature_path")
            if video_id is not None and feature_path is not None:
                self.video_to_feature_path[video_id] = feature_path
        self.all_video_ids = sorted(self.video_to_feature_path.keys())

    def __len__(self):
        return len(self.rows)

    def _resolve_feature_path(self, feature_path: str) -> Path:
        path = Path(feature_path)
        if not path.is_absolute():
            path = self.feature_dir / path
        return path

    def _load_features(self, feature_path: str):
        path = self._resolve_feature_path(feature_path)
        data = np.load(path, allow_pickle=True)
        features = data["features"].astype("float32")
        if self.max_frames is not None:
            features = features[: self.max_frames]
        return torch.from_numpy(features)

    def __getitem__(self, idx):
        row = self.rows[idx]
        neg_row = self.negatives_by_sample_id.get(row["sample_id"], {})
        requested_negative_ids = list(neg_row.get("negative_video_ids", []))
        positive_video_id = row["video_id"]
        if positive_video_id not in self.video_to_feature_path:
            raise KeyError(f"Positive video_id not found in manifest feature map: {positive_video_id}")

        negatives = []
        for video_id in requested_negative_ids:
            if video_id == positive_video_id:
                continue
            if video_id not in self.video_to_feature_path:
                continue
            if video_id in negatives:
                continue
            negatives.append(video_id)
            if len(negatives) >= self.shared_norm_num_negatives:
                break

        while len(negatives) < self.shared_norm_num_negatives and len(self.all_video_ids) > 1:
            video_id = random.choice(self.all_video_ids)
            if video_id == positive_video_id:
                continue
            if video_id in negatives:
                continue
            negatives.append(video_id)

        candidate_video_ids = [positive_video_id] + negatives[: self.shared_norm_num_negatives]
        candidate_features = []
        for video_id in candidate_video_ids:
            feature_path = row["feature_path"] if video_id == positive_video_id else self.video_to_feature_path[video_id]
            candidate_features.append(self._load_features(feature_path))

        gt_start_idx = int(row["gt_start_idx"])
        gt_end_idx = int(row["gt_end_idx"])
        if self.max_frames is not None and self.max_frames > 0:
            gt_start_idx = min(gt_start_idx, self.max_frames - 1)
            gt_end_idx = min(gt_end_idx, self.max_frames - 1)
            if gt_start_idx > gt_end_idx:
                gt_start_idx = gt_end_idx

        return {
            "sample_id": row["sample_id"],
            "query": row["query"],
            "candidate_video_ids": candidate_video_ids,
            "candidate_features": candidate_features,
            "positive_candidate_idx": 0,
            "gt_start_idx": gt_start_idx,
            "gt_end_idx": gt_end_idx,
        }


class LocalizerSharedNormCollator:
    def __init__(self, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(items)
        num_candidates = max(len(x["candidate_features"]) for x in items)
        max_n = max(feat.shape[0] for item in items for feat in item["candidate_features"])
        d_raw = items[0]["candidate_features"][0].shape[1]

        candidate_features = torch.zeros(batch_size, num_candidates, max_n, d_raw, dtype=torch.float32)
        candidate_feature_mask = torch.zeros(batch_size, num_candidates, max_n, dtype=torch.bool)

        sample_ids = []
        candidate_video_ids = []
        queries = []
        positive_candidate_idx = []
        gt_start_idx = []
        gt_end_idx = []

        for b, item in enumerate(items):
            sample_ids.append(item["sample_id"])
            candidate_video_ids.append(item["candidate_video_ids"])
            queries.append(item["query"])
            positive_candidate_idx.append(item["positive_candidate_idx"])
            gt_start_idx.append(item["gt_start_idx"])
            gt_end_idx.append(item["gt_end_idx"])

            for c, feat in enumerate(item["candidate_features"]):
                n = feat.shape[0]
                candidate_features[b, c, :n] = feat
                candidate_feature_mask[b, c, :n] = True

        toks = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "sample_ids": sample_ids,
            "candidate_video_ids": candidate_video_ids,
            "queries": queries,
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "candidate_features": candidate_features,
            "candidate_feature_mask": candidate_feature_mask,
            "positive_candidate_idx": torch.tensor(positive_candidate_idx, dtype=torch.long),
            "gt_start_idx": torch.tensor(gt_start_idx, dtype=torch.long),
            "gt_end_idx": torch.tensor(gt_end_idx, dtype=torch.long),
        }
