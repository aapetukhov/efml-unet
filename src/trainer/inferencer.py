from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import autocast
from tqdm.auto import tqdm

from src.metrics import MetricTracker
from src.utils import resolve_path, save_batch_images


class Inferencer:
    def __init__(
        self,
        model: torch.nn.Module,
        metrics,
        device: str,
        save_predictions: bool = False,
        predictions_dir: Optional[Path] = None,
        mixed_precision: bool = False,
        logger=None,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.device = device
        self.save_predictions = save_predictions
        self.predictions_dir = resolve_path(predictions_dir) if predictions_dir is not None else None
        self.mixed_precision = mixed_precision and device.startswith("cuda")
        self.logger = logger

    def _move_batch(self, batch):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def run(self, dataloader) -> Tuple[Dict[str, float], List[Path]]:
        tracker = MetricTracker(*[m.name for m in self.metrics])
        saved_paths: List[Path] = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="inference", leave=False):
                batch = self._move_batch(batch)
                lr, hr = batch["lr"], batch["hr"]
                with autocast(enabled=self.mixed_precision):
                    prediction = self.model(lr)

                batch_size = lr.size(0)
                for metric in self.metrics:
                    tracker.update(metric.name, float(metric(prediction.detach(), hr)), n=batch_size)

                if self.save_predictions and self.predictions_dir is not None:
                    saved_paths.extend(save_batch_images(prediction.cpu(), batch["name"], self.predictions_dir))

        return tracker.to_dict(), saved_paths
