import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets import get_dataloaders
from src.logger import get_logger
from src.trainer import Trainer
from src.utils import select_device, set_random_seed
from src.writer import create_writer

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline_x4")
def main(config):
    set_random_seed(config.seed)
    device = select_device(config.trainer.device)

    logger = get_logger("train")
    writer = create_writer(config, OmegaConf.to_container(config, resolve=True), logger)
    logger.info(f"Using device: {device}")

    dataloaders, _ = get_dataloaders(config, device)
    if "train" not in dataloaders:
        raise RuntimeError("Train dataloader is missing. Check dataset paths in config.")

    model     = instantiate(config.model).to(device)
    criterion = instantiate(config.criterion).to(device)
    optimizer = instantiate(config.optimization.optimizer, params=model.parameters())
    scheduler = instantiate(config.optimization.scheduler, optimizer=optimizer) \
        if getattr(config.optimization, "scheduler", None) else None

    metrics = {
        "train":     [instantiate(m) for m in config.metrics.train],
        "inference": [instantiate(m) for m in config.metrics.inference],
    }

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=dataloaders["train"],
        val_loader=dataloaders.get("val"),
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
