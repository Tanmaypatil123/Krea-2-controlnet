"""ControlNet-LoRA trainer for Krea-2 RAW — control-type agnostic.

Consumes latent shards produced by prepare_data.py (any control type: depth,
canny, tile, gray, or your own). The control latent is channel-concatenated to
the noisy latent; only the expanded input projection + LoRA weights train.
See README.md in this folder for the full recipe.
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

# repo root (k2_lora.py, mmdit.py) — this script lives in trainer/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from k2_lora import (
    K2_RAW_CONFIG,
    build_control_model,
    trainable_params,
    trainable_state_dict,
)
from mmdit import _mask, temb
from sampling import prepare, timesteps

# scheduler config: (base_image_seq_len, base_shift), (max_image_seq_len, max_shift)
MU_X1, MU_Y1, MU_X2, MU_Y2 = 256, 0.5, 6400, 1.15


def resolution_mu(seq_len: int) -> float:
    slope = (MU_Y2 - MU_Y1) / (MU_X2 - MU_X1)
    return slope * seq_len + (MU_Y1 - slope * MU_X1)


def shift_t(t: torch.Tensor, mu: float) -> torch.Tensor:
    return math.exp(mu) * t / (math.exp(mu) * t + 1.0 - t)


class LatentDataset(Dataset):
    def __init__(self, data_dir: str):
        self.records = []
        for idx_file in sorted(glob.glob(os.path.join(data_dir, "shard*", "index.jsonl"))):
            with open(idx_file) as f:
                for line in f:
                    self.records.append(json.loads(line))
        self.data_dir = data_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        d = np.load(os.path.join(self.data_dir, rec["file"]))
        return {
            "latent": torch.from_numpy(d["latent"].astype(np.float32)),
            "control": torch.from_numpy(d["control"].astype(np.float32)),
            "prompt": str(d["prompt"]),
        }


class BucketBatchSampler:
    def __init__(self, records, batch_size: int, seed: int = 0):
        self.by_bucket: dict[tuple, list[int]] = {}
        for i, rec in enumerate(records):
            self.by_bucket.setdefault(tuple(rec["bucket"]), []).append(i)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for idxs in self.by_bucket.values():
            idxs = idxs[:]
            rng.shuffle(idxs)
            for i in range(0, len(idxs) - self.batch_size + 1, self.batch_size):
                batches.append(idxs[i : i + self.batch_size])
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(len(v) // self.batch_size for v in self.by_bucket.values())


def collate(items):
    return {
        "latent": torch.stack([x["latent"] for x in items]),
        "control": torch.stack([x["control"] for x in items]),
        "prompts": [x["prompt"] for x in items],
    }


class TextConditioner(torch.nn.Module):
    """Qwen3-VL conditioner matching krea-2/encoder.py but with dynamic padding."""

    PREFIX = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    PREFIX_IDX = 34
    SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", max_length=512,
                 device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        self.qwen = (
            Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
            .to(device).eval().requires_grad_(False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_length = max_length
        self.device = device

    @torch.no_grad()
    def forward(self, prompts: list[str]):
        text = [self.PREFIX + p for p in prompts]
        inputs = self.tokenizer(
            text, truncation=True, padding="longest",
            max_length=self.max_length + self.PREFIX_IDX,
            return_tensors="pt", padding_side="right",
        ).to(self.device)
        suffix = self.tokenizer(
            [self.SUFFIX] * len(prompts), return_tensors="pt"
        ).to(self.device)
        input_ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1)
        mask = torch.cat(
            [inputs["attention_mask"].bool(), suffix["attention_mask"].bool()], dim=1
        )
        states = self.qwen(input_ids=input_ids, attention_mask=mask,
                           output_hidden_states=True)
        hiddens = torch.stack(
            [states.hidden_states[i] for i in self.SELECT_LAYERS], dim=2
        )
        return hiddens[:, self.PREFIX_IDX:], mask[:, self.PREFIX_IDX:]


def forward_control(model, img, ctrl, context, t, pos, mask, grad_ckpt=True):
    """SingleStreamDiT.forward with channel-concat control and gradient checkpointing."""
    x = model.first(torch.cat([img, ctrl], dim=-1))
    tv = model.tmlp(temb(t, model.config.tdim, device=x.device, dtype=x.dtype))
    tvec = model.tproj(tv)

    with torch.no_grad():
        txtmask = _mask(mask[:, : context.shape[1]])
        context = model.txtfusion(context, mask=txtmask)
        context = model.txtmlp(context)

    txtlen, imglen = context.shape[1], x.shape[1]
    combined = torch.cat((context, x), dim=1)

    padlen = (-combined.shape[1]) % 256
    if padlen > 0:
        combined = F.pad(combined, (0, 0, 0, padlen))
        mask = F.pad(mask, (0, padlen), value=False)
        pos = F.pad(pos, (0, 0, 0, padlen))

    mask = _mask(mask)
    freqs = model.posemb(pos)

    for block in model.blocks:
        if grad_ckpt:
            combined = checkpoint(block, combined, tvec, freqs, mask, use_reentrant=False)
        else:
            combined = block(combined, tvec, freqs, mask)

    final = model.last(combined, tv)
    return final[:, txtlen : txtlen + imglen, :]


@torch.no_grad()
def validation_images(model, vae_decode, conditioner, val_sample, device,
                      steps=28, cfg=3.5, seed=0):
    latent = val_sample["latent"][None].to(device)
    ctrl_lat = val_sample["control"][None].to(device, torch.bfloat16)
    prompt = val_sample["prompt"]
    patch = model.config.patch

    noise = torch.randn(
        latent.shape, device=device, dtype=torch.bfloat16,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    txt, txtmask = conditioner([prompt])
    untxt, untxtmask = conditioner([""])
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)
    ctrl, _, _ = prepare(ctrl_lat, txt.shape[1], patch, txtmask)

    ts = timesteps(x.shape[1], steps, MU_X1, MU_X2, y1=MU_Y1, y2=MU_Y2)
    img = x
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t = torch.full((1,), tcurr, dtype=img.dtype, device=device)
        cond = forward_control(model, img, ctrl, txt, t, pos, mask, grad_ckpt=False)
        uncond = forward_control(model, img, ctrl, untxt, t, unpos, unmask, grad_ckpt=False)
        v = cond + cfg * (cond - uncond)
        img = img + (tprev - tcurr) * v

    h, w = latent.shape[-2:]
    from einops import rearrange
    img = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    ph=patch, pw=patch, h=h // patch, w=w // patch)
    pixels = vae_decode(img.to(torch.bfloat16))
    pixels = (pixels.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    return pixels[0].permute(1, 2, 0).float().cpu().byte().numpy()


def upload_to_bucket(local_path: str, remote_rel: str, bucket_uri: str):
    cmd = ["hf", "buckets", "cp", local_path, f"{bucket_uri}/{remote_rel}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print(f"uploaded {remote_rel}")
    except subprocess.CalledProcessError as e:
        print(f"bucket upload FAILED for {remote_rel}: {e.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--ckpt-dir", default="/ckpts")
    ap.add_argument("--raw-ckpt", required=True, help="path to raw.safetensors")
    ap.add_argument("--control-type", default="depth",
                    help="label for run name / wandb / checkpoint metadata; "
                    "the control signal itself comes from the data")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0, help="debug: limit dataset size")
    ap.add_argument("--resume", default="", help="path to trainable-state safetensors; "
                    "also sets start step from filename/metadata")
    ap.add_argument("--caption-dropout", type=float, default=0.1)
    ap.add_argument("--wandb-project", default="krea2-controlnet")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--bucket-uri", default="",
                    help="optional hf bucket for checkpoint upload, "
                    "e.g. hf://buckets/user/controlnet-ckpts; empty = local only")
    args = ap.parse_args()

    device = "cuda"
    run_name = args.run_name or \
        f"{args.control_type}-lora-r{args.rank}-{time.strftime('%Y%m%d-%H%M')}"
    os.makedirs(os.path.join(args.ckpt_dir, run_name), exist_ok=True)

    ds = LatentDataset(args.data_dir)
    if args.max_samples:
        ds.records = ds.records[: args.max_samples]
    assert len(ds) > 0, f"no samples found under {args.data_dir}"
    print(f"dataset: {len(ds)} samples")

    sampler = BucketBatchSampler(ds.records, args.batch_size)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=8,
                        collate_fn=collate, pin_memory=True)

    print("building model...")
    model = build_control_model(args.raw_ckpt, rank=args.rank, device=device)
    start_step = 0
    if args.resume:
        from safetensors import safe_open
        from safetensors.torch import load_file as load_sd

        from k2_lora import load_trainable_state_dict
        load_trainable_state_dict(model, load_sd(args.resume))
        with safe_open(args.resume, framework="pt") as f:
            meta = f.metadata() or {}
        start_step = int(meta.get("step", 0))
        print(f"resumed from {args.resume} at step {start_step}")
    conditioner = TextConditioner(device=device)

    from diffusers import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image", subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval().requires_grad_(False)
    lat_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    lat_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    def vae_decode(z):
        z = (z.unsqueeze(2) * lat_std + lat_mean).to(torch.bfloat16)
        from einops import rearrange
        return rearrange(vae.decode(z).sample, "b c 1 h w -> b c h w")

    params = trainable_params(model)
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train/1e6:.1f}M")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)

    def lr_lambda(step):
        return min(1.0, (step + 1) / args.warmup)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_wandb = not args.no_wandb and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    val_sample = ds[0]
    patch = model.config.patch

    step = start_step
    t0 = time.time()
    loss_acc = 0.0
    model.train()

    data_iter = iter(loader)
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x0 = batch["latent"].to(device)
            ctrl_lat = batch["control"].to(device, torch.bfloat16)
            b = x0.shape[0]

            prompts = [p if random.random() >= args.caption_dropout else ""
                       for p in batch["prompts"]]
            txt, txtmask = conditioner(prompts)
            noise = torch.randn_like(x0)

            seq_len = (x0.shape[-2] // patch) * (x0.shape[-1] // patch)
            mu = resolution_mu(seq_len)
            u = torch.sigmoid(torch.randn(b, device=device))
            t = shift_t(u, mu)

            xt = t.view(-1, 1, 1, 1) * noise + (1 - t.view(-1, 1, 1, 1)) * x0
            target = noise - x0

            img, pos, mask = prepare(xt.to(torch.bfloat16), txt.shape[1], patch, txtmask)
            ctrl, _, _ = prepare(ctrl_lat, txt.shape[1], patch, txtmask)
            target_p, _, _ = prepare(target, txt.shape[1], patch, txtmask)

            pred = forward_control(model, img, ctrl, txt, t.to(torch.bfloat16), pos, mask)
            loss = F.mse_loss(pred.float(), target_p.float()) / args.grad_accum
            loss.backward()
            loss_acc += loss.item()

        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % args.log_every == 0:
            dt = (time.time() - t0) / args.log_every
            t0 = time.time()
            avg_loss = loss_acc / args.log_every
            loss_acc = 0.0
            print(f"step {step} | loss {avg_loss:.4f} | gnorm {gnorm:.2f} | {dt:.2f}s/step")
            if use_wandb:
                wandb.log({"loss": avg_loss, "grad_norm": gnorm.item(),
                           "lr": sched.get_last_lr()[0], "sec_per_step": dt}, step=step)

        if step % args.val_every == 0 and use_wandb:
            model.eval()
            img_np = validation_images(model, vae_decode, conditioner, val_sample, device)
            wandb.log({"val_image": wandb.Image(img_np, caption=val_sample["prompt"][:200])},
                      step=step)
            model.train()
            t0 = time.time()

        if step % args.save_every == 0 or step == args.max_steps:
            from safetensors.torch import save_file
            sd = trainable_state_dict(model)
            fname = f"step_{step:06d}.safetensors"
            local = os.path.join(args.ckpt_dir, run_name, fname)
            save_file(sd, local, metadata={"rank": str(args.rank), "step": str(step),
                                           "base": "krea/Krea-2-Raw",
                                           "control_type": args.control_type})
            if args.bucket_uri:
                upload_to_bucket(local, f"{run_name}/{fname}", args.bucket_uri)
            t0 = time.time()

    if use_wandb:
        wandb.finish()
    print("training done")


if __name__ == "__main__":
    main()
