from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .io_utils import JsonlReader


class FeatureManifestDataset(Dataset):
    def __init__(self, manifest_path: str, feature_dir: str, limit: Optional[int] = None):
        self.rows = JsonlReader.read(manifest_path, limit)
        self.feature_dir = Path(feature_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = self.feature_dir / feature_path
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
