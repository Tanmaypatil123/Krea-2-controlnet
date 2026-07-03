"""CLI for Krea-2 depth-ControlNet-LoRA inference.

Examples:
  # turbo (fast, recommended)
  python inference.py photo.jpg -p "a cozy cabin interior at dusk" \
      --lora step_006000.safetensors

  # raw base (undistilled, needs CFG)
  python inference.py photo.jpg -p "..." --lora step_006000.safetensors \
      --base raw --steps 28 --cfg 3.5

  # depth-only (no prompt)
  python inference.py photo.jpg --lora step_006000.safetensors
"""

import argparse
import os

from huggingface_hub import hf_hub_download
from PIL import Image

BASE_CKPTS = {"raw": ("krea/Krea-2-Raw", "raw.safetensors"),
              "turbo": ("krea/Krea-2-Turbo", "turbo.safetensors")}


def main():
    ap = argparse.ArgumentParser(description="Krea-2 depth LoRA inference")
    ap.add_argument("image", help="init image (depth source)")
    ap.add_argument("-p", "--prompt", default="", help="empty = depth-only")
    ap.add_argument("--lora", required=True, help="LoRA .safetensors path")
    ap.add_argument("--base", choices=["raw", "turbo"], default="turbo")
    ap.add_argument("--steps", type=int, default=None,
                    help="default: 8 turbo / 28 raw")
    ap.add_argument("--cfg", type=float, default=None,
                    help="default: 0 turbo / 3.5 raw")
    ap.add_argument("--mu", type=float, default=None,
                    help="timestep shift; default: 1.15 turbo / resolution-based raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-scale", type=float, default=1.0,
                    help="control strength dial (0.5 = weaker adherence)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("-o", "--output", default="output.png")
    ap.add_argument("--save-strip", action="store_true",
                    help="also save input|depth|output comparison strip")
    args = ap.parse_args()

    turbo = args.base == "turbo"
    steps = args.steps if args.steps is not None else (8 if turbo else 28)
    cfg = args.cfg if args.cfg is not None else (0.0 if turbo else 3.5)
    mu = args.mu if args.mu is not None else (1.15 if turbo else None)

    repo, fname = BASE_CKPTS[args.base]
    print(f"resolving {args.base} base checkpoint...")
    base_ckpt = os.path.realpath(hf_hub_download(repo, fname))

    from pipeline import DepthLoRAPipeline

    print("loading pipeline (13B DiT + Qwen3-VL-4B + VAE + DepthAnything)...")
    pipe = DepthLoRAPipeline(base_ckpt, args.lora, rank=args.rank,
                             lora_scale=args.lora_scale)

    image = Image.open(args.image)
    print(f"generating: {steps} steps, cfg {cfg}, seed {args.seed}")
    out, depth = pipe(image, args.prompt, steps=steps, cfg=cfg, mu=mu,
                      seed=args.seed)
    out.save(args.output)
    print(f"saved {args.output}")

    if args.save_strip:
        inp = image.convert("RGB").resize(
            (round(image.width * out.height / image.height), out.height))
        dep = depth.convert("RGB")
        strip = Image.new("RGB", (inp.width + dep.width + out.width, out.height))
        for x, im in [(0, inp), (inp.width, dep), (inp.width + dep.width, out)]:
            strip.paste(im, (x, 0))
        strip_path = os.path.splitext(args.output)[0] + "_strip.png"
        strip.save(strip_path)
        print(f"saved {strip_path}")


if __name__ == "__main__":
    main()
