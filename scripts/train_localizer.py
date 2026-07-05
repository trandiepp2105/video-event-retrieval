from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eventformer_v1_dynamic_tsm.config import TrainConfig
from eventformer_v1_dynamic_tsm.trainer_localizer import EventFormerLocalizerTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train EventFormer localizer.")
    parser.add_argument("--config", type=str, default="configs/train_config.json")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--retriever-checkpoint", type=str, default=None)
    parser.add_argument("--shared-norm-negatives", type=str, default=None)
    parser.add_argument("--text-model-path", type=str, default=None)
    parser.add_argument("--use-cross-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-event-auxiliary-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-shared-norm", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def build_config(args) -> TrainConfig:
    cfg = TrainConfig.from_json(args.config)
    cfg.train_stage = "localizer"
    cfg.use_moment_localizer = True
    for key in [
        "max_train_samples",
        "max_val_samples",
        "epochs",
        "batch_size",
        "device",
        "retriever_checkpoint",
        "shared_norm_negatives",
        "text_model_path",
        "use_cross_attention",
        "use_event_auxiliary_loss",
        "use_shared_norm",
    ]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    trainer = EventFormerLocalizerTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
