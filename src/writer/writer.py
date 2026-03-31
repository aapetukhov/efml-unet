from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from src.logger.wandb import init_wandb


@dataclass
class WriterConfig:
    name: str = "console"
    project: str = "efml"
    entity: Optional[str] = None
    run_name: str = "sr"


class BaseWriter:
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        raise NotImplementedError

    def add_image(self, tag: str, image, step: int) -> None:  # pragma: no cover - optional
        pass

    def close(self) -> None:
        pass


class ConsoleWriter(BaseWriter):
    def __init__(self, logger) -> None:
        self.logger = logger

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        prefix = f"{prefix}/" if prefix else ""
        pretty = ", ".join(f"{prefix}{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(f"step={step} | {pretty}")


class WandBWriter(BaseWriter):
    def __init__(self, config: WriterConfig, project_config: Dict[str, Any], logger) -> None:
        self.run = init_wandb(project=config.project, entity=config.entity, name=config.run_name, config=project_config)
        self.logger = logger

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        prefix = f"{prefix}/" if prefix else ""
        self.run.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


def create_writer(config, project_config: Dict[str, Any], logger):
    writer_cfg = WriterConfig(
        name=config.writer.name,
        project=config.writer.project,
        entity=getattr(config.writer, "entity", None),
        run_name=config.writer.run_name,
    )
    if writer_cfg.name == "wandb":
        return WandBWriter(writer_cfg, project_config, logger)
    return ConsoleWriter(logger)
