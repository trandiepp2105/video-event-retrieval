from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import TrainConfig, resolve_text_model_source
from .data import (
    BatchCollator,
    FeatureManifestDataset,
    LocalizerSharedNormCollator,
    LocalizerSharedNormDataset,
)
from .io_utils import JsonlReader, save_json, set_seed
from .localizer import EventAwareMomentLocalizer
from .metrics import span_iou
from .shared_norm import shared_norm_start_end_loss


class EventFormerLocalizerTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_source, local_only = resolve_text_model_source(cfg.text_model_name, cfg.text_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
        self.model: Optional[EventAwareMomentLocalizer] = None

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
        if self.cfg.use_shared_norm:
            if not self.cfg.shared_norm_negatives:
                raise ValueError("shared_norm_negatives must be set when use_shared_norm=True")
            collator = LocalizerSharedNormCollator(self.tokenizer, self.cfg.tokenizer_max_length)
            train_ds = LocalizerSharedNormDataset(
                manifest_path=self.cfg.train_manifest,
                feature_dir=self.cfg.feature_dir,
                negatives_path=self.cfg.shared_norm_negatives,
                shared_norm_num_negatives=self.cfg.shared_norm_num_negatives,
                max_frames=self.cfg.max_frames,
                limit=self.cfg.max_train_samples,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                collate_fn=collator,
            )
        else:
            collator = BatchCollator(self.tokenizer, self.cfg.tokenizer_max_length)
            train_ds = FeatureManifestDataset(self.cfg.train_manifest, self.cfg.feature_dir, self.cfg.max_train_samples)
            train_loader = DataLoader(
                train_ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                collate_fn=collator,
            )

        val_loader = None
        if self.cfg.val_manifest and Path(self.cfg.val_manifest).exists():
            val_collator = BatchCollator(self.tokenizer, self.cfg.tokenizer_max_length)
            val_ds = FeatureManifestDataset(self.cfg.val_manifest, self.cfg.feature_dir, self.cfg.max_val_samples)
            val_loader = DataLoader(
                val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                collate_fn=val_collator,
            )
        return train_loader, val_loader

    def build_model(self):
        d_raw = self._infer_d_raw()
        self.model = EventAwareMomentLocalizer(
            d_raw=d_raw,
            d_model=self.cfg.d_model,
            text_model_name=self.cfg.text_model_name,
            text_model_path=self.cfg.text_model_path,
            freeze_text_encoder=self.cfg.freeze_text_encoder,
            max_frames=self.cfg.max_frames,
            max_events=self.cfg.max_events,
            frame_layers=self.cfg.frame_layers,
            event_layers=self.cfg.event_layers,
            num_heads=self.cfg.num_heads,
            frame_anchor_sizes=self.cfg.frame_anchor_sizes,
            event_anchor_sizes=self.cfg.event_anchor_sizes,
            ff_dim=self.cfg.ff_dim,
            dropout=self.cfg.dropout,
            event_strategy=self.cfg.event_strategy,
            event_kmeans_num_events=self.cfg.event_kmeans_num_events,
            event_window_size=self.cfg.event_window_size,
            tsm_window_size=self.cfg.tsm_window_size,
            tsm_threshold_alpha=self.cfg.tsm_threshold_alpha,
            min_event_len=self.cfg.min_event_len,
            max_event_len=self.cfg.max_event_len,
            use_cross_attention=self.cfg.use_cross_attention,
            use_event_auxiliary_loss=self.cfg.use_event_auxiliary_loss,
            lambda_event_localizer=self.cfg.lambda_event_localizer,
            freeze_video_encoder_for_localizer=self.cfg.freeze_video_encoder_for_localizer,
        ).to(self.device)
        return self.model

    def load_retriever_checkpoint(self):
        if not self.cfg.retriever_checkpoint or self.model is None:
            return
        ckpt = torch.load(self.cfg.retriever_checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = self.model.video_encoder.load_state_dict(state, strict=False)
        print("Loaded retriever checkpoint into localizer video_encoder")
        print("missing keys:", missing[:20])
        print("unexpected keys:", unexpected[:20])

    def _move_batch(self, batch: Dict):
        moved = dict(batch)
        tensor_keys = [
            "features",
            "feature_mask",
            "input_ids",
            "attention_mask",
            "gt_start_idx",
            "gt_end_idx",
            "candidate_features",
            "candidate_feature_mask",
            "positive_candidate_idx",
        ]
        for key in tensor_keys:
            if key in moved:
                moved[key] = moved[key].to(self.device, non_blocking=True)
        return moved

    def _shared_norm_forward(self, batch: Dict):
        assert self.model is not None
        bsz, num_candidates, seq_len, d_raw = batch["candidate_features"].shape
        flat_features = batch["candidate_features"].view(bsz * num_candidates, seq_len, d_raw)
        flat_mask = batch["candidate_feature_mask"].view(bsz * num_candidates, seq_len)
        flat_input_ids = (
            batch["input_ids"]
            .unsqueeze(1)
            .expand(bsz, num_candidates, -1)
            .reshape(bsz * num_candidates, -1)
        )
        flat_attention_mask = (
            batch["attention_mask"]
            .unsqueeze(1)
            .expand(bsz, num_candidates, -1)
            .reshape(bsz * num_candidates, -1)
        )
        out = self.model(
            features=flat_features,
            feature_mask=flat_mask,
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            gt_start_idx=None,
            gt_end_idx=None,
        )
        start_logits = out["start_logits"].view(bsz, num_candidates, seq_len)
        end_logits = out["end_logits"].view(bsz, num_candidates, seq_len)
        loss = shared_norm_start_end_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            candidate_mask=batch["candidate_feature_mask"],
            positive_candidate_idx=batch["positive_candidate_idx"],
            gt_start_idx=batch["gt_start_idx"],
            gt_end_idx=batch["gt_end_idx"],
        )
        out["loss"] = loss
        out["start_logits_shared"] = start_logits
        out["end_logits_shared"] = end_logits
        return out

    def train_one_epoch(self, loader, optimizer, scaler, epoch: int):
        assert self.model is not None
        self.model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"localizer train epoch {epoch}")
        for batch in pbar:
            batch = self._move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.amp and self.device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                if self.cfg.use_shared_norm:
                    out = self._shared_norm_forward(batch)
                else:
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
        for batch in tqdm(loader, desc="localizer val"):
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

    @torch.inference_mode()
    def evaluate_svmr(self, loader):
        if loader is None:
            return {}
        assert self.model is not None
        self.model.eval()
        total = 0
        r1_iou03 = 0
        r1_iou05 = 0
        r1_iou07 = 0
        r5_iou05 = 0
        top1_iou_sum = 0.0
        predictions: List[Dict] = []

        for batch in tqdm(loader, desc="localizer svmr"):
            batch = self._move_batch(batch)
            out = self.model(
                features=batch["features"],
                feature_mask=batch["feature_mask"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                gt_start_idx=batch["gt_start_idx"],
                gt_end_idx=batch["gt_end_idx"],
            )
            top_spans = self.model.decode_top_spans(
                start_logits=out["start_logits"],
                end_logits=out["end_logits"],
                feature_mask=batch["feature_mask"],
                topk=5,
                max_span_len=self.cfg.max_localizer_span_len,
            )

            for b, spans in enumerate(top_spans):
                total += 1
                gt_span = (int(batch["gt_start_idx"][b].item()), int(batch["gt_end_idx"][b].item()))
                if not spans:
                    continue
                ious = [span_iou((span["start_idx"], span["end_idx"]), gt_span) for span in spans]
                top1_iou = ious[0]
                top1_iou_sum += top1_iou
                if top1_iou >= 0.3:
                    r1_iou03 += 1
                if top1_iou >= 0.5:
                    r1_iou05 += 1
                if top1_iou >= 0.7:
                    r1_iou07 += 1
                if max(ious) >= 0.5:
                    r5_iou05 += 1
                predictions.append({
                    "sample_id": batch["sample_ids"][b],
                    "video_id": batch["video_ids"][b],
                    "query": batch["queries"][b],
                    "gt_span": [gt_span[0], gt_span[1]],
                    "predictions": [
                        {
                            "start_idx": int(span["start_idx"]),
                            "end_idx": int(span["end_idx"]),
                            "score": float(span["score"]),
                            "iou": float(iou),
                        }
                        for span, iou in zip(spans, ious)
                    ],
                })

        denom = max(total, 1)
        return {
            "svmr_r1_iou_0_3": r1_iou03 / denom,
            "svmr_r1_iou_0_5": r1_iou05 / denom,
            "svmr_r1_iou_0_7": r1_iou07 / denom,
            "svmr_r5_iou_0_5": r5_iou05 / denom,
            "mean_top1_iou": top1_iou_sum / denom,
            "predictions": predictions,
        }

    def save_checkpoint(self, name: str, epoch: int, val_loss: Optional[float]):
        assert self.model is not None
        path = self.output_dir / name
        payload = {
            "epoch": epoch,
            "val_loss": val_loss,
            "train_config": asdict(self.cfg),
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)
        print("Saved", path)

    def _is_better(self, current, best):
        mode = getattr(self.cfg, "best_metric_mode", "max")
        if best is None:
            return True
        if mode == "min":
            return current < best
        if mode == "max":
            return current > best
        raise ValueError(f"Unknown best_metric_mode: {mode}")

    def fit(self):
        train_loader, val_loader = self.build_dataloaders()
        self.build_model()
        self.load_retriever_checkpoint()
        assert self.model is not None
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler() if self.cfg.amp and self.device.type == "cuda" else None
        best = None
        best_epoch = None
        log_rows = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, optimizer, scaler, epoch)
            val_loss = self.evaluate_loss(val_loader)
            val_metrics = self.evaluate_svmr(val_loader)
            prediction_rows = val_metrics.pop("predictions", [])
            log_row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
            log_rows.append(log_row)
            metric_name = getattr(self.cfg, "best_metric", "svmr_r1_iou_0_5")
            current_score = log_row[metric_name]
            val_loss_text = f"{float(val_loss):.4f}" if val_loss is not None else "nan"
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss_text} {metric_name}={current_score:.4f}")
            if not self.cfg.save_best_only:
                self.save_checkpoint("eventformer_localizer_last.pt", epoch, val_loss)
            if self._is_better(current_score, best):
                best = current_score
                best_epoch = epoch
                self.save_checkpoint("eventformer_localizer_best.pt", epoch, val_loss)
                save_json(prediction_rows, self.output_dir / "val_predictions_best.json")
        if self.cfg.save_best_only and log_rows:
            self.save_checkpoint("eventformer_localizer_last.pt", self.cfg.epochs, log_rows[-1]["val_loss"])
        if best is not None and best_epoch is not None:
            print(f"Best metric: {self.cfg.best_metric}={best:.4f} at epoch {best_epoch}")
        save_json(log_rows, self.output_dir / "train_log_localizer.json")
        save_json(asdict(self.cfg), self.output_dir / "train_config.json")
        return log_rows
