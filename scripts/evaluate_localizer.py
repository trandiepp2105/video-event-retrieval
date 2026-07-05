from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from eventformer_v1_dynamic_tsm.config import TrainConfig
from eventformer_v1_dynamic_tsm.io_utils import write_jsonl
from eventformer_v1_dynamic_tsm.trainer_localizer import EventFormerLocalizerTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EventFormer localizer.")
    parser.add_argument("--config", type=str, default="configs/train_config.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig.from_json(args.config)
    cfg.train_stage = "localizer"
    cfg.use_moment_localizer = True
    trainer = EventFormerLocalizerTrainer(cfg)
    _, val_loader = trainer.build_dataloaders()
    trainer.build_model()
    assert trainer.model is not None

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.to(trainer.device)

    metrics = trainer.evaluate_svmr(val_loader)
    predictions = metrics.pop("predictions", [])
    write_jsonl(predictions, Path(args.output))
    print(metrics)


if __name__ == "__main__":
    main()
