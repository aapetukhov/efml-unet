import warnings

import hydra
import torch
from omegaconf import OmegaConf

from src.datasets import get_dataloaders
from src.logger import get_logger
from src.loss import L1LossWrapper
from src.metrics import SSIMMetric
from src.model import build_model
from src.trainer import Trainer
from src.utils import select_device, set_random_seed
from src.writer import create_writer

warnings.filterwarnings("ignore", category=UserWarning)


def build_metrics(config, section: str):
    metric_map = {
        "ssim": SSIMMetric,
    }
    metrics = []
    for metric_cfg in getattr(config.metrics, section, []):
        name = metric_cfg.name if hasattr(metric_cfg, "name") else metric_cfg.get("name")
        if name not in metric_map:
            continue
        metrics.append(metric_map[name]())
    return metrics


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    set_random_seed(config.seed)
    device = select_device(config.trainer.device)

    project_config = OmegaConf.to_container(config, resolve=True)
    logger = get_logger("train")
    writer = create_writer(config, project_config, logger)
    logger.info(f"Using device: {device}")

    dataloaders, _ = get_dataloaders(config, device)
    if "train" not in dataloaders:
        raise RuntimeError("Train dataloader is missing. Check dataset paths in config.")
    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val")

    model = build_model(config).to(device)
    criterion = L1LossWrapper().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = None
    if getattr(config.optimization, "scheduler", None) and config.optimization.scheduler.name == "cosine":
        t_max = (
            config.optimization.scheduler.T_max
            if config.optimization.scheduler.T_max is not None
            else config.trainer.epochs
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    metrics = {
        "train": build_metrics(config, "train"),
        "inference": build_metrics(config, "inference"),
    }

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=metrics,
        config=config,
        device=device,
        logger=logger,
        writer=writer,
    )

    trainer.train()
    writer.close()


if __name__ == "__main__":
    main()
