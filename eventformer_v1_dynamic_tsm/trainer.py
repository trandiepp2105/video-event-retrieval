from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from transformers import AutoTokenizer
except Exception as e:
    raise ImportError("Please install transformers") from e

from .config import TrainConfig
from .data import BatchCollator, FeatureManifestDataset
from .io_utils import JsonlReader, save_json, set_seed
from .model import EventFormerV1DynamicTSM


class EventFormerTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
        self.model: Optional[EventFormerV1DynamicTSM] = None

    def _infer_d_raw(self) -> int:
        rows = JsonlReader.read(self.cfg.train_manifest, limit=1)
        if not rows:
            raise RuntimeError("Empty train manifest")
        feature_path = Path(rows[0]["feature_path"])
        if not feature_path.is_absolute():
            feature_path = Path(self.cfg.feature_dir) / feature_path
        data = np.load(feature_path, allow_pickle=True)
        return int(data["features"].shape[1])

    def build_dataloaders(self):
        collator = BatchCollator(self.tokenizer, self.cfg.tokenizer_max_length)
        train_ds = FeatureManifestDataset(self.cfg.train_manifest, self.cfg.feature_dir, self.cfg.max_train_samples)
        val_ds = None
        if self.cfg.val_manifest and Path(self.cfg.val_manifest).exists():
            val_ds = FeatureManifestDataset(self.cfg.val_manifest, self.cfg.feature_dir, self.cfg.max_val_samples)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collator,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                collate_fn=collator,
            )
        return train_loader, val_loader

    def build_model(self):
        d_raw = self._infer_d_raw()
        self.model = EventFormerV1DynamicTSM(
            d_raw=d_raw,
            d_model=self.cfg.d_model,
            text_model_name=self.cfg.text_model_name,
            freeze_text_encoder=self.cfg.freeze_text_encoder,
            query_pooling=self.cfg.query_pooling,
            use_modality_specific_query=self.cfg.use_modality_specific_query,
            modalities=tuple(self.cfg.modalities),
            max_frames=self.cfg.max_frames,
            max_events=self.cfg.max_events,
            frame_layers=self.cfg.frame_layers,
            event_layers=self.cfg.event_layers,
            num_heads=self.cfg.num_heads,
            frame_anchor_sizes=list(self.cfg.frame_anchor_sizes),
            event_anchor_sizes=list(self.cfg.event_anchor_sizes),
            ff_dim=self.cfg.ff_dim,
            dropout=self.cfg.dropout,
            tsm_window_size=self.cfg.tsm_window_size,
            tsm_threshold_alpha=self.cfg.tsm_threshold_alpha,
            min_event_len=self.cfg.min_event_len,
            max_event_len=self.cfg.max_event_len,
            lambda_event=self.cfg.lambda_event,
            use_hard_negative=self.cfg.use_hard_negative,
            lambda_hard=self.cfg.lambda_hard,
            use_weak_positive=self.cfg.use_weak_positive,
            lambda_weak=self.cfg.lambda_weak,
            lambda_weak_event=self.cfg.lambda_weak_event,
            weak_positive_margin=self.cfg.weak_positive_margin,
            temperature=self.cfg.temperature,
        ).to(self.device)
        return self.model

    def _move_batch(self, batch):
        keys = ["features", "feature_mask", "input_ids", "attention_mask", "gt_start_idx", "gt_end_idx"]
        for k in keys:
            batch[k] = batch[k].to(self.device, non_blocking=True)
        return batch

    def train_one_epoch(self, loader, optimizer, scaler, epoch: int):
        assert self.model is not None
        self.model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"train epoch {epoch}")
        for batch in pbar:
            batch = self._move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.amp and self.device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                out = self.model(
                    features=batch["features"],
                    feature_mask=batch["feature_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    gt_start_idx=batch["gt_start_idx"],
                    gt_end_idx=batch["gt_end_idx"],
                )
                loss = out["loss"]
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                optimizer.step()
            total += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))
        return total / max(1, len(loader))

    @torch.inference_mode()
    def evaluate_loss(self, loader):
        if loader is None:
            return None
        assert self.model is not None
        self.model.eval()
        total = 0.0
        for batch in tqdm(loader, desc="val"):
            batch = self._move_batch(batch)
            out = self.model(
                features=batch["features"],
                feature_mask=batch["feature_mask"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                gt_start_idx=batch["gt_start_idx"],
                gt_end_idx=batch["gt_end_idx"],
            )
            total += float(out["loss"].item())
        return total / max(1, len(loader))

    def save_checkpoint(self, name: str, epoch: int, val_loss: Optional[float]):
        assert self.model is not None
        path = self.output_dir / name
        payload = {
            "epoch": epoch,
            "val_loss": val_loss,
            "train_config": asdict(self.cfg),
            "model_config": self.model.config_dict,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)
        print("Saved", path)

    def run(self):
        train_loader, val_loader = self.build_dataloaders()
        self.build_model()
        assert self.model is not None
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler() if self.cfg.amp and self.device.type == "cuda" else None
        best = float("inf")
        log_rows = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, optimizer, scaler, epoch)
            val_loss = self.evaluate_loss(val_loader)
            log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(log_rows[-1])
            if not self.cfg.save_best_only:
                self.save_checkpoint("eventformer_v1_last.pt", epoch, val_loss)
            score = val_loss if val_loss is not None else train_loss
            if score < best:
                best = score
                self.save_checkpoint("eventformer_v1_best.pt", epoch, val_loss)
        if self.cfg.save_best_only:
            self.save_checkpoint("eventformer_v1_last.pt", epoch=self.cfg.epochs, val_loss=log_rows[-1]["val_loss"] if log_rows else None)
        save_json(log_rows, self.output_dir / "train_log.json")
        save_json(asdict(self.cfg), self.output_dir / "train_config.json")
        return log_rows
