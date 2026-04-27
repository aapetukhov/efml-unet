from __future__ import annotations

from typing import Dict, List

import torch


def sr_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    lr = torch.stack([sample["lr"] for sample in batch])
    hr = torch.stack([sample["hr"] for sample in batch])
    names = [sample["name"] for sample in batch]
    scales = [sample["scale"] for sample in batch]
    return {"lr": lr, "hr": hr, "name": names, "scale": scales}
