from __future__ import annotations

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


def init_wandb(**kwargs):
    if wandb is None:
        raise ImportError("wandb is not installed. Install it or switch writer to 'console'.")
    return wandb.init(**kwargs)
