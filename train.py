from __future__ import annotations

import argparse

from eventformer_v1_dynamic_tsm.config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train EventFormer V1 Dynamic TSM from manifests.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-stage", type=str, choices=["retriever", "localizer"], default=None)
    parser.add_argument("--retriever-checkpoint", type=str, default=None)
    parser.add_argument("--shared-norm-negatives", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text-model-name", type=str, default=None)
    parser.add_argument("--text-model-path", type=str, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--frame-layers", type=int, default=None)
    parser.add_argument("--event-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    parser.add_argument("--query-pooling", type=str, default=None)
    parser.add_argument("--event-strategy", type=str, default=None)
    parser.add_argument("--event-kmeans-num-events", type=int, default=None)
    parser.add_argument("--event-window-size", type=int, default=None)
    parser.add_argument("--event-stride", type=int, default=None)
    parser.add_argument("--event-window-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--event-stride-ratio", type=float, default=None)
    parser.add_argument("--event-pooling", type=str, default=None)
    parser.add_argument("--lambda-event-localizer", type=float, default=None)
    parser.add_argument("--max-localizer-span-len", type=int, default=None)
    parser.add_argument("--shared-norm-num-negatives", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lambda-frame", type=float, default=None)
    parser.add_argument("--lambda-event", type=float, default=None)
    parser.add_argument("--weak-positive-weight", type=float, default=None)
    parser.add_argument("--lambda-hard", type=float, default=None)
    parser.add_argument("--lambda-weak", type=float, default=None)
    parser.add_argument("--lambda-weak-event", type=float, default=None)
    parser.add_argument("--weak-positive-margin", type=int, default=None)
    parser.add_argument("--tokenizer-max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--best-metric", type=str, default=None)
    parser.add_argument("--best-metric-mode", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--freeze-text-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-modality-specific-query", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-hard-negative", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-weak-positive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-moment-localizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-cross-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-event-auxiliary-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-video-encoder-for-localizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-shared-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-best-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def build_config(args) -> TrainConfig:
    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    for key in [
        "train_manifest",
        "val_manifest",
        "feature_dir",
        "output_dir",
        "train_stage",
        "retriever_checkpoint",
        "shared_norm_negatives",
        "batch_size",
        "epochs",
        "num_workers",
        "device",
        "text_model_name",
        "text_model_path",
        "d_model",
        "max_frames",
        "max_events",
        "frame_layers",
        "event_layers",
        "num_heads",
        "ff_dim",
        "query_pooling",
        "event_strategy",
        "event_kmeans_num_events",
        "event_window_size",
        "event_stride",
        "event_window_sizes",
        "event_stride_ratio",
        "event_pooling",
        "lambda_event_localizer",
        "max_localizer_span_len",
        "shared_norm_num_negatives",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "temperature",
        "lambda_frame",
        "lambda_event",
        "weak_positive_weight",
        "lambda_hard",
        "lambda_weak",
        "lambda_weak_event",
        "weak_positive_margin",
        "tokenizer_max_length",
        "seed",
        "best_metric",
        "best_metric_mode",
        "max_train_samples",
        "max_val_samples",
        "freeze_text_encoder",
        "use_modality_specific_query",
        "use_hard_negative",
        "use_weak_positive",
        "use_moment_localizer",
        "use_cross_attention",
        "use_event_auxiliary_loss",
        "freeze_video_encoder_for_localizer",
        "use_shared_norm",
        "amp",
        "save_best_only",
        "normalize_embeddings",
    ]:
        value = getattr(args, key)
        if value is not None:
            if key == "event_window_sizes":
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    if cfg.train_stage == "localizer" or cfg.use_moment_localizer:
        from eventformer_v1_dynamic_tsm.trainer_localizer import EventFormerLocalizerTrainer

        trainer = EventFormerLocalizerTrainer(cfg)
        log_rows = trainer.fit()
    else:
        from eventformer_v1_dynamic_tsm.trainer import EventFormerTrainer

        trainer = EventFormerTrainer(cfg)
        log_rows = trainer.run()
    if log_rows:
        print("Last epoch:", log_rows[-1])


if __name__ == "__main__":
    main()
