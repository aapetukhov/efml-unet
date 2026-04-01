from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.metrics import MetricTracker
from src.utils import prepare_checkpoint_path, save_checkpoint


class BaseTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        metrics: Dict[str, list],
        config,
        device: str,
        logger,
        writer,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics
        self.config = config
        self.device = device
        self.logger = logger
        self.writer = writer
        self.grad_clip = config.trainer.grad_clip
        self.val_every = config.trainer.val_every
        self.epochs = config.trainer.epochs
        self.scaler = GradScaler(device="cuda", enabled=config.trainer.mixed_precision and device.startswith("cuda"))
        self.global_step = 0
        self.checkpoint_path = prepare_checkpoint_path(config)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _compute_metrics(self, prediction: torch.Tensor, target: torch.Tensor, stage: str) -> Dict[str, float]:
        values = {}
        for metric in self.metrics.get(stage, []):
            # prediction, target: (B, C, H, W). Compute mean over batch.
            batch_vals = []
            for pred_i, targ_i in zip(prediction, target):
                batch_vals.append(metric(pred_i, targ_i))
            values[metric.name] = float(sum(batch_vals) / max(len(batch_vals), 1))
        return values

    def _train_val_epoch(self, dataloader, training: bool, epoch: int) -> Dict[str, float]:
        mode = "train" if training else "val"
        tracker = MetricTracker("loss", *[m.name for m in self.metrics.get("train" if training else "inference", [])])
        self.model.train(training)

        progress = tqdm(dataloader, desc=f"{mode} {epoch}", leave=False)
        for batch in progress:
            self.global_step += 1 if training else 0
            batch = self._move_batch(batch)
            lr = batch["lr"]
            hr = batch["hr"]

            with autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu", enabled=self.scaler.is_enabled() and training):
                prediction = self.model(lr)
                losses = self.criterion(prediction, hr)
                loss = losses["loss"]

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

            batch_size = lr.size(0)
            tracker.update("loss", loss.item(), n=batch_size)
            metric_set = "train" if training else "inference"
            for name, value in self._compute_metrics(prediction.detach(), hr, stage=metric_set).items():
                tracker.update(name, float(value), n=batch_size)

            if training and self.global_step % self.config.trainer.log_every == 0:
                self.writer.log_metrics(tracker.to_dict(), step=self.global_step, prefix=mode)

        return tracker.to_dict()

    def _save_checkpoint(self, epoch: int, extra: Optional[Dict[str, float]] = None) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if extra:
            payload.update(extra)
        save_checkpoint(self.checkpoint_path, payload)
        self.logger.info(f"Saved checkpoint to {self.checkpoint_path}")

    def train(self):
        best_ssim = -1.0
        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_val_epoch(self.train_loader, training=True, epoch=epoch)
            self.logger.info(f"Epoch {epoch} train: {train_metrics}")

            val_metrics = None
            if self.val_loader is not None and epoch % self.val_every == 0:
                with torch.no_grad():
                    val_metrics = self._train_val_epoch(self.val_loader, training=False, epoch=epoch)
                self.logger.info(f"Epoch {epoch} val: {val_metrics}")
                current_ssim = val_metrics.get("ssim", -1.0)
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    self._save_checkpoint(epoch, extra={"best_ssim": best_ssim})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        if best_ssim < 0:
            # still save the latest weights
            self._save_checkpoint(self.epochs)
