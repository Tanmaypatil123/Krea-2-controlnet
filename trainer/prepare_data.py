"""Preprocess an image+caption dataset into latent shards for Krea-2 ControlNet-LoRA training.

For every sample: pick the nearest ~1MP aspect bucket, extract a control image
(depth / canny / tile / gray — or a user-supplied paired image), VAE-encode the
image and the control with the Qwen-Image VAE, and write one .npz per sample
plus an index.jsonl per shard. See README.md for the on-disk format the trainer
expects.

Input modes:
  --source folder  local directory of images; captions from `<name>.txt`
                   sidecars or a metadata.jsonl with {"file_name", "text"}
  --source hf      stream parquet shards of a HF image dataset
                   (dataset id, shard count and column names configurable)
"""

import argparse
import glob
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# (w, h), area ~= 1024^2, multiples of 16
BUCKETS = [
    (1024, 1024),
    (896, 1152), (1152, 896),
    (832, 1216), (1216, 832),
    (768, 1344), (1344, 768),
    (704, 1472), (1472, 704),
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def pick_bucket(w: int, h: int) -> tuple[int, int]:
    ar = w / h
    return min(BUCKETS, key=lambda b: abs(np.log(b[0] / b[1]) - np.log(ar)))


def resize_center_crop(img: Image.Image, tw: int, th: int) -> Image.Image:
    w, h = img.size
    scale = max(tw / w, th / h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    w, h = img.size
    left, top = (w - tw) // 2, (h - th) // 2
    return img.crop((left, top, left + tw, top + th))


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> (3, h, w) float in [-1, 1]."""
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32))
    return x.permute(2, 0, 1) / 127.5 - 1.0


class VAEEncoder(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        self.ae = AutoencoderKLQwenImage.from_pretrained(
            "Qwen/Qwen-Image", subfolder="vae", torch_dtype=dtype
        ).to(device).eval().requires_grad_(False)
        self.mean = torch.tensor(self.ae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(self.ae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (b, 3, h, w) in [-1, 1] -> normalized latent (b, 16, h/8, w/8)"""
        x = x.unsqueeze(2)
        z = self.ae.encode(x).latent_dist.sample()
        z = (z - self.mean) / self.std
        return z.squeeze(2)


# ---------------------------------------------------------------------------
# Control extractors: list[PIL RGB] -> list[(3, h, w) float tensor in [-1, 1]]
# each output the same size as its input image. To add a control type, write a
# class with this __call__ signature and register it in build_extractor().
# ---------------------------------------------------------------------------


class DepthControl:
    """Depth-Anything-V2-Large inverse depth (near = white)."""

    def __init__(self, device="cuda", dtype=torch.float16):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = "depth-anything/Depth-Anything-V2-Large-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype)
            .to(device).eval().requires_grad_(False)
        )
        self.device = device

    @torch.no_grad()
    def __call__(self, images: list[Image.Image]) -> list[torch.Tensor]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        depth = self.model(**inputs).predicted_depth  # (b, H, W)
        out = []
        for d, img in zip(depth, images):
            d = d[None, None].float()
            d = F.interpolate(d, size=(img.height, img.width), mode="bilinear", align_corners=False)
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            out.append(d[0].repeat(3, 1, 1) * 2.0 - 1.0)
        return out


class CannyControl:
    """OpenCV Canny edges, white on black."""

    def __init__(self, low=100, high=200):
        import cv2

        self.cv2, self.low, self.high = cv2, low, high

    def __call__(self, images: list[Image.Image]) -> list[torch.Tensor]:
        out = []
        for img in images:
            edges = self.cv2.Canny(np.asarray(img.convert("L")), self.low, self.high)
            e = torch.from_numpy(edges).float() / 255.0
            out.append(e[None].repeat(3, 1, 1) * 2.0 - 1.0)
        return out


class TileControl:
    """Downscale by `factor` and upscale back — blur control for tile/upscaling."""

    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, images: list[Image.Image]) -> list[torch.Tensor]:
        out = []
        for img in images:
            x = to_tensor(img)[None]
            small = F.interpolate(
                x, size=(max(1, img.height // self.factor), max(1, img.width // self.factor)),
                mode="bilinear", align_corners=False,
            )
            out.append(F.interpolate(small, size=(img.height, img.width),
                                     mode="bilinear", align_corners=False)[0])
        return out


class GrayControl:
    """Grayscale — recolorization control."""

    def __call__(self, images: list[Image.Image]) -> list[torch.Tensor]:
        out = []
        for img in images:
            g = torch.from_numpy(np.asarray(img.convert("L"), dtype=np.float32))
            out.append(g[None].repeat(3, 1, 1) / 127.5 - 1.0)
        return out


def build_extractor(args):
    if args.control_type == "depth":
        return DepthControl()
    if args.control_type == "canny":
        return CannyControl(args.canny_low, args.canny_high)
    if args.control_type == "tile":
        return TileControl(args.tile_factor)
    if args.control_type == "gray":
        return GrayControl()
    if args.control_type == "custom":
        return None  # paired control images are loaded per-sample instead
    raise ValueError(f"unknown control type {args.control_type}")


# ---------------------------------------------------------------------------
# Shared encode/write path. `items` are dicts with keys:
#   img (bucket-cropped PIL), prompt (str), id (str),
#   ctrl_img (optional bucket-cropped PIL, used instead of the extractor)
# ---------------------------------------------------------------------------


def encode_group(grouped, vae, extractor, shard_name, shard_out, index, batch_size, device):
    n = 0
    for (bw, bh), items in grouped.items():
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            pil_imgs = [c["img"] for c in chunk]

            if chunk[0].get("ctrl_img") is not None:
                ctrls = [to_tensor(c["ctrl_img"]) for c in chunk]
            else:
                ctrls = extractor(pil_imgs)

            imgs = torch.stack([to_tensor(im) for im in pil_imgs]).to(device)
            ctrl_t = torch.stack(ctrls).to(device)

            lat = vae.encode(imgs.to(torch.bfloat16))
            ctrl = vae.encode(ctrl_t.to(torch.bfloat16))

            for j, c in enumerate(chunk):
                name = f"{c['id']}.npz"
                np.savez_compressed(
                    os.path.join(shard_out, name),
                    latent=lat[j].float().cpu().numpy().astype(np.float16),
                    control=ctrl[j].float().cpu().numpy().astype(np.float16),
                    prompt=np.str_(c["prompt"]),
                    size=np.array([bw, bh]),
                )
                index.append({"file": f"{shard_name}/{name}", "bucket": [bw, bh]})
                n += 1
    return n


def write_index(shard_out, index):
    with open(os.path.join(shard_out, "index.jsonl"), "w") as f:
        for rec in index:
            f.write(json.dumps(rec) + "\n")
    open(os.path.join(shard_out, "_DONE"), "w").close()


# ---------------------------------------------------------------------------
# folder mode
# ---------------------------------------------------------------------------


def load_captions(input_dir: str) -> dict[str, str]:
    meta = os.path.join(input_dir, "metadata.jsonl")
    captions = {}
    if os.path.exists(meta):
        with open(meta) as f:
            for line in f:
                rec = json.loads(line)
                captions[rec["file_name"]] = rec.get("text", "")
    return captions


def find_paired_control(control_dir: str, stem: str) -> str:
    for ext in IMAGE_EXTS:
        p = os.path.join(control_dir, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no control image for '{stem}' in {control_dir}")


def process_folder(args, device="cuda"):
    files = sorted(
        f for f in glob.glob(os.path.join(args.input_dir, "*"))
        if f.lower().endswith(IMAGE_EXTS)
    )
    assert files, f"no images found in {args.input_dir}"
    captions = load_captions(args.input_dir)
    print(f"{len(files)} images, {len(captions)} captions from metadata.jsonl")

    vae = VAEEncoder(device=device)
    extractor = build_extractor(args)

    shards = [files[i : i + args.samples_per_shard]
              for i in range(0, len(files), args.samples_per_shard)]
    for shard_idx, shard_files in enumerate(shards):
        shard_name = f"shard{shard_idx:05d}"
        shard_out = os.path.join(args.out_dir, shard_name)
        if os.path.exists(os.path.join(shard_out, "_DONE")):
            print(f"{shard_name}: already done, skipping")
            continue
        os.makedirs(shard_out, exist_ok=True)

        index, n = [], 0
        for start in range(0, len(shard_files), 64):
            grouped: dict[tuple[int, int], list[dict]] = {}
            for path in shard_files[start : start + 64]:
                img = Image.open(path).convert("RGB")
                bucket = pick_bucket(*img.size)
                fname = os.path.basename(path)
                stem = os.path.splitext(fname)[0]
                prompt = captions.get(fname, "")
                if not prompt:
                    txt = os.path.join(args.input_dir, stem + ".txt")
                    if os.path.exists(txt):
                        prompt = open(txt, encoding="utf-8").read().strip()
                item = {"img": resize_center_crop(img, *bucket), "prompt": prompt, "id": stem}
                if args.control_type == "custom":
                    ctrl = Image.open(find_paired_control(args.control_dir, stem)).convert("RGB")
                    item["ctrl_img"] = resize_center_crop(ctrl, *bucket)
                grouped.setdefault(bucket, []).append(item)

            n += encode_group(grouped, vae, extractor, shard_name, shard_out,
                              index, args.batch_size, device)

        write_index(shard_out, index)
        print(f"{shard_name}: {n} samples done")


# ---------------------------------------------------------------------------
# hf mode (streams parquet shards; one process can handle a shard range, so
# preprocessing parallelizes across workers/containers by --shards)
# ---------------------------------------------------------------------------


def load_shard(dataset: str, shard_idx: int, num_shards: int, token: str | None):
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("hf", token=token)
    path = f"datasets/{dataset}/data/train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
    return pq.ParquetFile(fs.open(path))


def decode_image_cell(cell) -> Image.Image:
    if isinstance(cell, dict):
        cell = cell["bytes"]
    return Image.open(io.BytesIO(cell)).convert("RGB")


def process_shards(shard_indices, out_dir, args, device="cuda"):
    token = os.environ.get("HF_TOKEN")
    vae = VAEEncoder(device=device)
    extractor = build_extractor(args)

    for shard_idx in shard_indices:
        shard_name = f"shard{shard_idx:05d}"
        shard_out = os.path.join(out_dir, shard_name)
        if os.path.exists(os.path.join(shard_out, "_DONE")):
            print(f"{shard_name}: already done, skipping")
            continue
        os.makedirs(shard_out, exist_ok=True)

        pf = load_shard(args.dataset, shard_idx, args.num_shards, token)
        index, n, row_idx = [], 0, 0
        for batch in pf.iter_batches(batch_size=64, columns=[args.prompt_col, args.image_col]):
            rows = batch.to_pylist()
            # group by bucket so VAE/extractor batches share a shape
            grouped: dict[tuple[int, int], list[dict]] = {}
            for row in rows:
                img = decode_image_cell(row[args.image_col])
                bucket = pick_bucket(*img.size)
                grouped.setdefault(bucket, []).append({
                    "img": resize_center_crop(img, *bucket),
                    "prompt": row[args.prompt_col],
                    "id": f"{shard_idx:05d}_{row_idx:06d}",
                })
                row_idx += 1

            n += encode_group(grouped, vae, extractor, shard_name, shard_out,
                              index, args.batch_size, device)

        write_index(shard_out, index)
        print(f"{shard_name}: {n} samples done")


def parse_shard_spec(spec: str, num_shards: int) -> list[int]:
    if spec == "all":
        return list(range(num_shards))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["folder", "hf"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--control-type", default="depth",
                    choices=["depth", "canny", "tile", "gray", "custom"])
    ap.add_argument("--batch-size", type=int, default=8)
    # folder mode
    ap.add_argument("--input-dir", help="folder: directory of images (+ captions)")
    ap.add_argument("--control-dir", help="folder + custom: paired control images, "
                    "same filename stem as the source image")
    ap.add_argument("--samples-per-shard", type=int, default=1000)
    # hf mode
    ap.add_argument("--dataset", help="hf: dataset id, e.g. user/name")
    ap.add_argument("--num-shards", type=int, help="hf: total parquet shard count")
    ap.add_argument("--shards", default="all", help="hf: e.g. '0-9' or '3,7' for parallel workers")
    ap.add_argument("--image-col", default="images")
    ap.add_argument("--prompt-col", default="prompts")
    # extractor knobs
    ap.add_argument("--canny-low", type=int, default=100)
    ap.add_argument("--canny-high", type=int, default=200)
    ap.add_argument("--tile-factor", type=int, default=8)
    args = ap.parse_args()

    if args.source == "folder":
        assert args.input_dir, "--input-dir required with --source folder"
        assert args.control_type != "custom" or args.control_dir, \
            "--control-dir required with --control-type custom"
        process_folder(args)
    else:
        assert args.dataset and args.num_shards, \
            "--dataset and --num-shards required with --source hf"
        assert args.control_type != "custom", \
            "custom control needs paired local files; use --source folder"
        process_shards(parse_shard_spec(args.shards, args.num_shards), args.out_dir, args)


if __name__ == "__main__":
    main()
