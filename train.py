from __future__ import annotations

import argparse

from video_event_retrieval.eventformer_v1_dynamic_tsm.config import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train EventFormer V1 Dynamic TSM from manifests.")
    parser.add_argument(
        "--config",
        type=str,
        default="video_event_retrieval/configs/train_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument("--train-manifest", type=str, default=None)
    parser.add_argument("--val-manifest", type=str, default=None)
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--text-model-name", type=str, default=None)
    parser.add_argument("--query-pooling", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--lambda-event", type=float, default=None)
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-best-only", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def build_config(args) -> TrainConfig:
    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    for key in [
        "train_manifest",
        "val_manifest",
        "feature_dir",
        "output_dir",
        "batch_size",
        "epochs",
        "num_workers",
        "device",
        "text_model_name",
        "query_pooling",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "lambda_event",
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
        "amp",
        "save_best_only",
    ]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    from video_event_retrieval.eventformer_v1_dynamic_tsm.trainer import EventFormerTrainer

    trainer = EventFormerTrainer(cfg)
    log_rows = trainer.run()
    if log_rows:
        print("Last epoch:", log_rows[-1])


if __name__ == "__main__":
    main()
