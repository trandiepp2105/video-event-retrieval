from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import TrainConfig, resolve_text_model_source
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
        tokenizer_source, local_only = resolve_text_model_source(cfg.text_model_name, cfg.text_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
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
            text_model_path=self.cfg.text_model_path,
            freeze_text_encoder=self.cfg.freeze_text_encoder,
            query_pooling=self.cfg.query_pooling,
            query_transformer_layers=self.cfg.query_transformer_layers,
            query_transformer_heads=self.cfg.query_transformer_heads,
            query_transformer_ff_dim=self.cfg.query_transformer_ff_dim,
            query_transformer_dropout=self.cfg.query_transformer_dropout,
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
            event_strategy=self.cfg.event_strategy,
            event_kmeans_num_events=self.cfg.event_kmeans_num_events,
            event_window_size=self.cfg.event_window_size,
            event_stride=self.cfg.event_stride,
            event_window_sizes=tuple(self.cfg.event_window_sizes),
            event_stride_ratio=self.cfg.event_stride_ratio,
            event_pooling=self.cfg.event_pooling,
            tsm_window_size=self.cfg.tsm_window_size,
            tsm_threshold_alpha=self.cfg.tsm_threshold_alpha,
            min_event_len=self.cfg.min_event_len,
            max_event_len=self.cfg.max_event_len,
            normalize_embeddings=self.cfg.normalize_embeddings,
            lambda_frame=self.cfg.lambda_frame,
            lambda_event=self.cfg.lambda_event,
            weak_positive_weight=self.cfg.weak_positive_weight,
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
        moved = dict(batch)
        keys = ["features", "feature_mask", "input_ids", "attention_mask", "gt_start_idx", "gt_end_idx"]
        for k in keys:
            moved[k] = moved[k].to(self.device, non_blocking=True)
        return moved

    def _build_scheduler(self, optimizer, steps_per_epoch: int):
        if self.cfg.lr_scheduler in (None, "none"):
            return None
        total_steps = max(1, steps_per_epoch * self.cfg.epochs)
        warmup_steps = min(total_steps - 1, max(0, int(total_steps * self.cfg.warmup_ratio)))
        min_lr_ratio = float(self.cfg.min_lr_ratio)

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            if self.cfg.lr_scheduler == "linear":
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
            if self.cfg.lr_scheduler == "cosine":
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
            raise ValueError(f"Unsupported lr_scheduler: {self.cfg.lr_scheduler}")

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def train_one_epoch(self, loader, optimizer, scaler, scheduler, epoch: int):
        assert self.model is not None
        self.model.train()
        total = 0.0
        total_events = 0.0
        total_batches_with_events = 0
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
                    video_ids=batch.get("video_ids", None),
                )
                loss = out["loss"]
                loss_frame = out.get("loss_frame")
                loss_event = out.get("loss_event")
                event_spans = out.get("event_spans")
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
            if scheduler is not None:
                scheduler.step()
            total += float(loss.item())
            postfix = {"loss": f"{float(loss.item()):.4f}"}
            if loss_frame is not None:
                postfix["l_frame"] = f"{float(loss_frame.item()):.4f}"
            if loss_event is not None:
                postfix["l_event"] = f"{float(loss_event.item()):.4f}"
            if event_spans:
                num_events = [len(spans) for spans in event_spans]
                batch_ev_mean = sum(num_events) / max(1, len(num_events))
                total_events += batch_ev_mean
                total_batches_with_events += 1
                postfix["ev_mean"] = f"{batch_ev_mean:.1f}"
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            pbar.set_postfix(postfix)
        return {
            "loss": total / max(1, len(loader)),
            "train_event_mean": total_events / max(1, total_batches_with_events),
            "lr": optimizer.param_groups[0]["lr"],
        }

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
                video_ids=batch.get("video_ids", None),
            )
            total += float(out["loss"].item())
        return total / max(1, len(loader))

    def _load_video_features(self, dataset: FeatureManifestDataset, feature_path: str) -> torch.Tensor:
        resolved = dataset._resolve_feature_path(feature_path)
        data = np.load(resolved, allow_pickle=True)
        features = data["features"].astype("float32")
        if self.cfg.max_frames is not None:
            features = features[: self.cfg.max_frames]
        return torch.from_numpy(features)

    def _build_retrieval_corpus(self, dataset: FeatureManifestDataset):
        assert self.model is not None

        unique_rows = {}
        for row in dataset.rows:
            video_id = row["video_id"]
            if video_id not in unique_rows:
                unique_rows[video_id] = row

        video_ids = list(unique_rows.keys())
        if not video_ids:
            return {
                "video_ids": [],
                "video_id_to_index": {},
                "frame_embeddings": torch.empty(0, self.model.d_model),
                "frame_video_indices": torch.empty(0, dtype=torch.long),
                "event_embeddings": torch.empty(0, self.model.d_model),
                "event_video_indices": torch.empty(0, dtype=torch.long),
                "frame_count_mean": 0.0,
                "event_count_mean": 0.0,
            }

        flat_frame_embeddings = []
        flat_frame_video_indices = []
        flat_event_embeddings = []
        flat_event_video_indices = []
        frame_counts = []
        event_counts = []
        encode_batch_size = max(1, min(self.cfg.batch_size, 32))

        for start in tqdm(range(0, len(video_ids), encode_batch_size), desc="build_vr_corpus"):
            batch_video_ids = video_ids[start : start + encode_batch_size]
            feature_list = [self._load_video_features(dataset, unique_rows[video_id]["feature_path"]) for video_id in batch_video_ids]
            max_len = max(int(feat.shape[0]) for feat in feature_list)
            d_raw = int(feature_list[0].shape[1])

            features = torch.zeros(len(feature_list), max_len, d_raw, dtype=torch.float32)
            feature_mask = torch.zeros(len(feature_list), max_len, dtype=torch.bool)
            for idx, feat in enumerate(feature_list):
                cur_len = int(feat.shape[0])
                if cur_len <= 0:
                    continue
                features[idx, :cur_len] = feat
                feature_mask[idx, :cur_len] = True

            encoded = self.model.encode_video(
                features=features.to(self.device, non_blocking=True),
                feature_mask=feature_mask.to(self.device, non_blocking=True),
                normalize=True,
            )
            event_embeddings = encoded["event_embeddings"].detach().cpu()
            event_mask = encoded["event_mask"].detach().cpu()
            frame_embeddings = encoded["frame_embeddings"].detach().cpu()
            frame_mask = encoded["frame_mask"].detach().cpu()

            for local_idx, _video_id in enumerate(batch_video_ids):
                valid_frames = frame_mask[local_idx].bool()
                cur_frame_embeddings = frame_embeddings[local_idx, valid_frames]
                frame_counts.append(int(cur_frame_embeddings.shape[0]))
                if cur_frame_embeddings.shape[0] > 0:
                    global_video_idx = start + local_idx
                    flat_frame_embeddings.append(cur_frame_embeddings)
                    flat_frame_video_indices.append(
                        torch.full((cur_frame_embeddings.shape[0],), global_video_idx, dtype=torch.long)
                    )

                valid_events = event_mask[local_idx].bool()
                cur_embeddings = event_embeddings[local_idx, valid_events]
                if cur_embeddings.shape[0] == 0:
                    cur_embeddings = cur_frame_embeddings
                event_counts.append(int(cur_embeddings.shape[0]))
                if cur_embeddings.shape[0] == 0:
                    continue
                global_video_idx = start + local_idx
                flat_event_embeddings.append(cur_embeddings)
                flat_event_video_indices.append(
                    torch.full((cur_embeddings.shape[0],), global_video_idx, dtype=torch.long)
                )

        if flat_frame_embeddings:
            frame_embeddings = torch.cat(flat_frame_embeddings, dim=0)
            frame_video_indices = torch.cat(flat_frame_video_indices, dim=0)
        else:
            frame_embeddings = torch.empty(0, self.model.d_model)
            frame_video_indices = torch.empty(0, dtype=torch.long)

        if flat_event_embeddings:
            event_embeddings = torch.cat(flat_event_embeddings, dim=0)
            event_video_indices = torch.cat(flat_event_video_indices, dim=0)
        else:
            event_embeddings = torch.empty(0, self.model.d_model)
            event_video_indices = torch.empty(0, dtype=torch.long)

        return {
            "video_ids": video_ids,
            "video_id_to_index": {video_id: idx for idx, video_id in enumerate(video_ids)},
            "frame_embeddings": frame_embeddings,
            "frame_video_indices": frame_video_indices,
            "event_embeddings": event_embeddings,
            "event_video_indices": event_video_indices,
            "frame_count_mean": (sum(frame_counts) / len(frame_counts)) if frame_counts else 0.0,
            "event_count_mean": (sum(event_counts) / len(event_counts)) if event_counts else 0.0,
        }

    @torch.inference_mode()
    def evaluate_retrieval(self, loader):
        if loader is None:
            return {}
        assert self.model is not None
        self.model.eval()

        dataset = getattr(loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "rows"):
            return {}

        corpus = self._build_retrieval_corpus(dataset)
        num_videos = len(corpus["video_ids"])
        if num_videos == 0:
            return {}

        total = 0
        recall_hits_frame = {1: 0, 5: 0, 10: 0, 100: 0}
        recall_hits_event = {1: 0, 5: 0, 10: 0, 100: 0}
        recall_hits_combined = {1: 0, 5: 0, 10: 0, 100: 0}
        rank_sum_frame = 0.0
        rank_sum_event = 0.0
        rank_sum_combined = 0.0
        frame_embeddings = corpus["frame_embeddings"]
        frame_video_indices = corpus["frame_video_indices"]
        event_embeddings = corpus["event_embeddings"]
        event_video_indices = corpus["event_video_indices"]
        score_chunk_size = 16384

        for batch in tqdm(loader, desc="val_vr"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            q = self.model.encode_query(
                input_ids=input_ids,
                attention_mask=attention_mask,
                normalize=True,
            )
            batch_size = int(q.shape[0])
            total += batch_size

            neg_inf = -1e4
            frame_retrieval_scores = q.new_full((batch_size, num_videos), neg_inf)
            event_retrieval_scores = q.new_full((batch_size, num_videos), neg_inf)

            if frame_embeddings.numel() > 0:
                for start in range(0, frame_embeddings.shape[0], score_chunk_size):
                    end = min(start + score_chunk_size, frame_embeddings.shape[0])
                    emb_chunk = frame_embeddings[start:end].to(self.device, non_blocking=True)
                    score_chunk = q @ emb_chunk.t()
                    video_idx_chunk = frame_video_indices[start:end].to(self.device, non_blocking=True)
                    scatter_index = video_idx_chunk.unsqueeze(0).expand(batch_size, -1)
                    if hasattr(frame_retrieval_scores, "scatter_reduce_"):
                        frame_retrieval_scores.scatter_reduce_(1, scatter_index, score_chunk, reduce="amax", include_self=True)
                    else:
                        for local_video_idx in torch.unique(video_idx_chunk).tolist():
                            mask = video_idx_chunk == local_video_idx
                            frame_retrieval_scores[:, local_video_idx] = torch.maximum(
                                frame_retrieval_scores[:, local_video_idx],
                                score_chunk[:, mask].max(dim=1).values,
                            )

            if event_embeddings.numel() > 0:
                for start in range(0, event_embeddings.shape[0], score_chunk_size):
                    end = min(start + score_chunk_size, event_embeddings.shape[0])
                    emb_chunk = event_embeddings[start:end].to(self.device, non_blocking=True)
                    score_chunk = q @ emb_chunk.t()
                    video_idx_chunk = event_video_indices[start:end].to(self.device, non_blocking=True)
                    scatter_index = video_idx_chunk.unsqueeze(0).expand(batch_size, -1)
                    if hasattr(event_retrieval_scores, "scatter_reduce_"):
                        event_retrieval_scores.scatter_reduce_(1, scatter_index, score_chunk, reduce="amax", include_self=True)
                    else:
                        for local_video_idx in torch.unique(video_idx_chunk).tolist():
                            mask = video_idx_chunk == local_video_idx
                            event_retrieval_scores[:, local_video_idx] = torch.maximum(
                                event_retrieval_scores[:, local_video_idx],
                                score_chunk[:, mask].max(dim=1).values,
                            )

            combined_retrieval_scores = torch.maximum(frame_retrieval_scores, event_retrieval_scores)

            target_video_indices = torch.tensor(
                [corpus["video_id_to_index"][video_id] for video_id in batch["video_ids"]],
                dtype=torch.long,
                device=self.device,
            )
            score_sets = (
                ("frame", frame_retrieval_scores, recall_hits_frame),
                ("event", event_retrieval_scores, recall_hits_event),
                ("combined", combined_retrieval_scores, recall_hits_combined),
            )
            for name, retrieval_scores, recall_hits in score_sets:
                sorted_video_indices = torch.argsort(retrieval_scores, dim=1, descending=True)
                matches = sorted_video_indices.eq(target_video_indices.unsqueeze(1))
                target_ranks = matches.float().argmax(dim=1) + 1
                if name == "frame":
                    rank_sum_frame += float(target_ranks.sum().item())
                elif name == "event":
                    rank_sum_event += float(target_ranks.sum().item())
                else:
                    rank_sum_combined += float(target_ranks.sum().item())

                for k in recall_hits:
                    topk = min(k, num_videos)
                    recall_hits[k] += int(matches[:, :topk].any(dim=1).sum().item())

        denom = max(total, 1)
        return {
            "vr_frame_r1": recall_hits_frame[1] / denom,
            "vr_frame_r5": recall_hits_frame[5] / denom,
            "vr_frame_r10": recall_hits_frame[10] / denom,
            "vr_frame_r100": recall_hits_frame[100] / denom,
            "vr_frame_mean_rank": rank_sum_frame / denom,
            "vr_event_r1": recall_hits_event[1] / denom,
            "vr_event_r5": recall_hits_event[5] / denom,
            "vr_event_r10": recall_hits_event[10] / denom,
            "vr_event_r100": recall_hits_event[100] / denom,
            "vr_event_mean_rank": rank_sum_event / denom,
            "vr_r1": recall_hits_combined[1] / denom,
            "vr_r5": recall_hits_combined[5] / denom,
            "vr_r10": recall_hits_combined[10] / denom,
            "vr_r100": recall_hits_combined[100] / denom,
            "vr_mean_rank": rank_sum_combined / denom,
            "vr_num_videos": num_videos,
            "vr_mean_frames_per_video": corpus["frame_count_mean"],
            "vr_mean_events_per_video": corpus["event_count_mean"],
        }

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
        text_encoder_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("text_encoder."):
                text_encoder_params.append(param)
            else:
                other_params.append(param)

        optimizer_groups = []
        if other_params:
            optimizer_groups.append(
                {
                    "params": other_params,
                    "lr": self.cfg.lr,
                    "weight_decay": self.cfg.weight_decay,
                }
            )
        if text_encoder_params:
            optimizer_groups.append(
                {
                    "params": text_encoder_params,
                    "lr": self.cfg.text_encoder_lr if self.cfg.text_encoder_lr is not None else self.cfg.lr,
                    "weight_decay": self.cfg.weight_decay,
                }
            )
        optimizer = torch.optim.AdamW(optimizer_groups)
        scheduler = self._build_scheduler(optimizer, steps_per_epoch=len(train_loader))
        scaler = torch.amp.GradScaler("cuda") if self.cfg.amp and self.device.type == "cuda" else None
        if self.cfg.best_metric_mode == "min":
            best = float("inf")
        else:
            best = float("-inf")
        log_rows = []
        for epoch in range(1, self.cfg.epochs + 1):
            train_stats = self.train_one_epoch(train_loader, optimizer, scaler, scheduler, epoch)
            train_loss = train_stats["loss"]
            val_loss = self.evaluate_loss(val_loader)
            val_metrics = self.evaluate_retrieval(val_loader)
            log_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_event_mean": train_stats["train_event_mean"],
                "lr": train_stats["lr"],
                "val_loss": val_loss,
                **val_metrics,
            }
            log_rows.append(log_row)
            print(log_row)
            if not self.cfg.save_best_only:
                self.save_checkpoint("eventformer_v1_last.pt", epoch, val_loss)
            score = log_row.get(self.cfg.best_metric)
            if score is None:
                score = val_loss if val_loss is not None else train_loss
            is_better = score < best if self.cfg.best_metric_mode == "min" else score > best
            if is_better:
                best = score
                self.save_checkpoint("eventformer_v1_best.pt", epoch, val_loss)
        if self.cfg.save_best_only:
            self.save_checkpoint("eventformer_v1_last.pt", epoch=self.cfg.epochs, val_loss=log_rows[-1]["val_loss"] if log_rows else None)
        save_json(log_rows, self.output_dir / "train_log.json")
        save_json(asdict(self.cfg), self.output_dir / "train_config.json")
        return log_rows
