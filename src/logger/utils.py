from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def tensor_to_figure(tensor: torch.Tensor, title: str = ""):
    image = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
