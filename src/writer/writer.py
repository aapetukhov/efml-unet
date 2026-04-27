from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torchvision.utils as vutils

from src.logger.wandb import init_wandb


@dataclass
class WriterConfig:
    name: str = "console"
    project: str = "efml"
    entity: Optional[str] = None
    run_name: str = "sr"
    log_dir: str = "runs"


class BaseWriter:
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        raise NotImplementedError

    def add_images(
        self,
        tag: str,
        lr: torch.Tensor,
        sr: torch.Tensor,
        hr: torch.Tensor,
        step: int,
        max_samples: int = 4,
    ) -> None:
        pass  # no-op for writers that don't support images

    def close(self) -> None:
        pass


class ConsoleWriter(BaseWriter):
    def __init__(self, logger) -> None:
        self.logger = logger

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        prefix = f"{prefix}/" if prefix else ""
        pretty = ", ".join(f"{prefix}{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"step={step} | {pretty}")


class TensorBoardWriter(BaseWriter):
    def __init__(self, log_dir: str, logger) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter as TBWriter
        except ImportError:
            raise ImportError("tensorboard is not installed. Run: pip install tensorboard")
        self.tb = TBWriter(log_dir=log_dir)
        self.logger = logger
        self.logger.info(f"TensorBoard: writing to '{log_dir}'")
        self.logger.info(f"  on server:  tensorboard --logdir={log_dir}")
        self.logger.info(f"  SSH tunnel: ssh -L 6006:localhost:6006 <user@remote>")
        self.logger.info(f"  then open:  http://localhost:6006")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        prefix = f"{prefix}/" if prefix else ""
        for k, v in metrics.items():
            self.tb.add_scalar(f"{prefix}{k}", v, step)

    def add_images(self, tag, lr, sr, hr, step, max_samples=4):
        n = min(lr.size(0), max_samples)
        # 3 rows: LR input | SR prediction | HR ground truth
        grid = vutils.make_grid(
            torch.cat([lr[:n], sr[:n].clamp(0.0, 1.0), hr[:n]], dim=0),
            nrow=n,
            normalize=False,
            padding=2,
        )
        self.tb.add_image(tag, grid, step)

    def close(self) -> None:
        self.tb.close()


class WandBWriter(BaseWriter):
    def __init__(self, config: WriterConfig, project_config: Dict[str, Any], logger) -> None:
        self.run = init_wandb(
            project=config.project,
            entity=config.entity,
            name=config.run_name,
            config=project_config,
        )
        self.logger = logger

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        prefix = f"{prefix}/" if prefix else ""
        self.run.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def add_images(self, tag, lr, sr, hr, step, max_samples=4):
        import wandb
        n = min(lr.size(0), max_samples)
        panels = []
        for i in range(n):
            panels.extend([
                wandb.Image(lr[i].permute(1, 2, 0).numpy(), caption=f"[{i}] LR input"),
                wandb.Image(sr[i].clamp(0.0, 1.0).permute(1, 2, 0).numpy(), caption=f"[{i}] SR output"),
                wandb.Image(hr[i].permute(1, 2, 0).numpy(), caption=f"[{i}] HR target"),
            ])
        self.run.log({tag: panels}, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


def create_writer(config, project_config: Dict[str, Any], logger):
    name = config.writer.name

    if name == "tensorboard":
        log_dir = getattr(config.writer, "log_dir", f"runs/{config.experiment_name}")
        return TensorBoardWriter(log_dir=log_dir, logger=logger)

    if name == "wandb":
        writer_cfg = WriterConfig(
            name=name,
            project=getattr(config.writer, "project", "efml"),
            entity=getattr(config.writer, "entity", None),
            run_name=getattr(config.writer, "run_name", config.experiment_name),
        )
        return WandBWriter(writer_cfg, project_config, logger)

    return ConsoleWriter(logger)
