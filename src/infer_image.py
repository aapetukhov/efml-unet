from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.config import load_config
from src.modeling import build_model, prepare_model_for_inference


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    image = TF.to_pil_image(tensor.clamp(0.0, 1.0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SR inference on a single LR image (with optional HR for comparison)."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the LR input image.",
    )
    parser.add_argument(
        "--hr", type=str, default=None,
        help="Optional path to the HR ground-truth image for side-by-side comparison.",
    )
    parser.add_argument("--output-dir", type=str, default="./results/inference_examples")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["benchmark"]["device"])
    scale = config["data"]["scale"]

    model = build_model(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
        scale=scale,
    )
    checkpoint = torch.load(config["train"]["save_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = prepare_model_for_inference(
        model=model,
        device=device,
        use_fp16=config["benchmark"]["use_fp16"],
    )

    lr_image = Image.open(args.input).convert("RGB")
    lr_tensor = TF.to_tensor(lr_image).unsqueeze(0).to(device)
    if config["benchmark"]["use_fp16"] and device.type == "cuda":
        lr_tensor = lr_tensor.half()

    with torch.inference_mode():
        prediction = model(lr_tensor).float().clamp(0.0, 1.0).cpu()[0]

    lr_w, lr_h = lr_image.size
    bicubic_image = lr_image.resize((lr_w * scale, lr_h * scale), Image.BICUBIC)
    bicubic_tensor = TF.to_tensor(bicubic_image)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    stem = input_path.stem

    save_tensor_image(TF.to_tensor(lr_image), output_dir / f"{stem}_lr.png")
    save_tensor_image(bicubic_tensor, output_dir / f"{stem}_bicubic.png")
    save_tensor_image(prediction, output_dir / f"{stem}_prediction.png")

    print(f"Scale: x{scale}")
    print(f"Saved LR input      → {output_dir / f'{stem}_lr.png'}")
    print(f"Saved bicubic        → {output_dir / f'{stem}_bicubic.png'}")
    print(f"Saved prediction     → {output_dir / f'{stem}_prediction.png'}")

    if args.hr is not None:
        hr_image = Image.open(args.hr).convert("RGB")
        hr_tensor = TF.to_tensor(hr_image)
        save_tensor_image(hr_tensor, output_dir / f"{stem}_hr.png")
        print(f"Saved HR ground truth → {output_dir / f'{stem}_hr.png'}")


if __name__ == "__main__":
    main()
