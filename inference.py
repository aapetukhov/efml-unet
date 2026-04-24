import warnings
from pathlib import Path

import hydra
from hydra.utils import instantiate

from src.datasets import get_dataloaders
from src.logger import get_logger
from src.trainer import Inferencer
from src.utils import load_checkpoint, resolve_path, select_device, set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.seed)
    device = select_device(config.device)
    logger = get_logger("inference")

    dataloaders, _ = get_dataloaders(config, device)
    loader = dataloaders.get("test") or dataloaders.get("val")
    if loader is None:
        raise RuntimeError("No dataloader found. Ensure datasets.test or datasets.val is set.")

    model = instantiate(config.model).to(device)
    checkpoint = load_checkpoint(resolve_path(config.checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint: {config.checkpoint_path}")

    metrics = [instantiate(m) for m in config.metrics.inference]
    inferencer = Inferencer(
        model=model,
        metrics=metrics,
        device=device,
        save_predictions=config.save_predictions,
        predictions_dir=Path(config.predictions_dir),
        mixed_precision=getattr(config, "mixed_precision", False),
        logger=logger,
    )
    metrics_dict, saved = inferencer.run(loader)
    logger.info(f"Inference metrics: {metrics_dict}")
    if saved:
        logger.info(f"Saved {len(saved)} predictions to {config.predictions_dir}")


if __name__ == "__main__":
    main()
