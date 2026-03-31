import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.datasets import get_dataloaders
from src.logger import get_logger
from src.metrics import PSNRMetric, SSIMMetric
from src.model import build_model
from src.trainer import Inferencer
from src.utils import load_checkpoint, resolve_path, select_device, set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


def build_metrics(config):
    metric_map = {
        "psnr": PSNRMetric,
        "ssim": SSIMMetric,
    }
    metrics = []
    for metric_cfg in getattr(config.metrics, "inference", []):
        name = metric_cfg.name if hasattr(metric_cfg, "name") else metric_cfg.get("name")
        if name not in metric_map:
            continue
        metrics.append(metric_map[name]())
    return metrics


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.seed)
    device = select_device(config.device)
    logger = get_logger("inference")

    dataloaders, _ = get_dataloaders(config, device)
    loader = dataloaders.get("test") or dataloaders.get("val")
    if loader is None:
        raise RuntimeError("No dataloader found for inference. Ensure datasets.test or datasets.val is set.")

    model = build_model(config).to(device)
    checkpoint_path = resolve_path(config.checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    metrics = build_metrics(config)
    inferencer = Inferencer(
        model=model,
        metrics=metrics,
        device=device,
        save_predictions=config.save_predictions,
        predictions_dir=Path(config.predictions_dir),
        mixed_precision=config.get("mixed_precision", False),
        logger=logger,
    )
    metrics_dict, saved = inferencer.run(loader)
    logger.info(f"Inference metrics: {metrics_dict}")
    if saved:
        logger.info(f"Saved {len(saved)} prediction images to {config.predictions_dir}")


if __name__ == "__main__":
    main()
