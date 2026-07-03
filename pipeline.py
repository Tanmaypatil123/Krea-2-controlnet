"""Standalone depth-ControlNet-LoRA pipeline for Krea-2.

Everything needed for inference: LoRA surgery, text conditioning, VAE,
depth estimation, and the flow-matching sampler with control injection.
`mmdit.py` is the unmodified DiT definition from the official krea-2 repo.
"""

import math
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file

from mmdit import SingleMMDiTConfig, SingleStreamDiT, _mask, temb

K2_CONFIG = SingleMMDiTConfig(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12, multiplier=4,
    layers=28, patch=2, channels=16, txtheads=20, txtkvheads=20, txtlayers=12,
)

BASE_CKPTS = {"raw": ("krea/Krea-2-Raw", "raw.safetensors"),
              "turbo": ("krea/Krea-2-Turbo", "turbo.safetensors")}

LORA_TARGETS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
                "mlp.gate", "mlp.up", "mlp.down")

# timestep-shift interpolation endpoints (image seq len -> mu), from the
# scheduler config Krea-2 was trained with
MU_X1, MU_Y1, MU_X2, MU_Y2 = 256, 0.5, 6400, 1.15

BUCKETS = [(1024, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832),
           (768, 1344), (1344, 768), (704, 1472), (1472, 704)]


# ---------------------------------------------------------------- model surgery

class LoRALinear(nn.Module):
    """y = Wx + scale * B(Ax). A: rank x in, B: out x rank."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.scale = (alpha / rank) * scale
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

    def forward(self, x):
        lora = (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)
        return self.base(x) + lora * self.scale


class ControlInputLayer(nn.Module):
    """Replaces the DiT input projection: in_features doubled (64 -> 128) to
    accept [noisy latent patches ; depth latent patches] concatenated on the
    channel dim. Trained weights are loaded from the LoRA checkpoint."""

    def __init__(self, pretrained: nn.Linear):
        super().__init__()
        in_f, out_f = pretrained.in_features, pretrained.out_features
        self.weight = nn.Parameter(torch.zeros(out_f, in_f * 2, dtype=torch.float32))
        self.bias = nn.Parameter(pretrained.bias.detach().float().clone())
        with torch.no_grad():
            self.weight[:, :in_f] = pretrained.weight.detach().float()

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))


def _get(root, path):
    for p in path.split("."):
        root = getattr(root, p)
    return root


def _set(root, path, new):
    parts = path.split(".")
    setattr(_get(root, ".".join(parts[:-1])) if len(parts) > 1 else root,
            parts[-1], new)


def build_model(base_ckpt: str, lora_ckpt: str, rank: int = 64,
                lora_scale: float = 1.0, device: str = "cuda",
                dtype: torch.dtype = torch.bfloat16) -> SingleStreamDiT:
    with torch.device("meta"):
        model = SingleStreamDiT(K2_CONFIG)
    model.load_state_dict(load_file(base_ckpt), strict=True, assign=True)
    model = model.to(device=device, dtype=dtype).requires_grad_(False)

    model.first = ControlInputLayer(model.first).to(device)
    for i in range(K2_CONFIG.layers):
        for t in LORA_TARGETS:
            path = f"blocks.{i}.{t}"
            _set(model, path, LoRALinear(_get(model, path), rank, rank,
                                         lora_scale).to(device))

    sd = load_file(lora_ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return model.eval()


# ---------------------------------------------------------------- conditioning

class TextConditioner(nn.Module):
    """Qwen3-VL-4B encoder exactly as Krea-2 uses it: hidden states from 12
    selected layers, stacked, with the chat-template prefix sliced off."""

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

        self.qwen = (Qwen3VLForConditionalGeneration
                     .from_pretrained(model_id, torch_dtype=dtype)
                     .to(device).eval().requires_grad_(False))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_length = max_length
        self.device = device

    @torch.no_grad()
    def forward(self, prompts):
        text = [self.PREFIX + p for p in prompts]
        inputs = self.tokenizer(
            text, truncation=True, padding="longest",
            max_length=self.max_length + self.PREFIX_IDX,
            return_tensors="pt", padding_side="right").to(self.device)
        suffix = self.tokenizer([self.SUFFIX] * len(prompts),
                                return_tensors="pt").to(self.device)
        ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(),
                          suffix["attention_mask"].bool()], dim=1)
        states = self.qwen(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)
        hiddens = torch.stack([states.hidden_states[i]
                               for i in self.SELECT_LAYERS], dim=2)
        return hiddens[:, self.PREFIX_IDX:], mask[:, self.PREFIX_IDX:]


class VAE(nn.Module):
    """Qwen-Image VAE (f8, 16ch) with Krea-2's latent normalization."""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        self.ae = (AutoencoderKLQwenImage
                   .from_pretrained("Qwen/Qwen-Image", subfolder="vae",
                                    torch_dtype=dtype)
                   .to(device).eval().requires_grad_(False))
        self.mean = torch.tensor(self.ae.config.latents_mean,
                                 device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(self.ae.config.latents_std,
                                device=device).view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def encode(self, x):  # (b,3,h,w) in [-1,1] -> (b,16,h/8,w/8) normalized
        z = self.ae.encode(x.unsqueeze(2)).latent_dist.sample()
        return ((z - self.mean) / self.std).squeeze(2)

    @torch.no_grad()
    def decode(self, z):  # normalized latent -> (b,3,h,w) in [-1,1]
        z = (z.unsqueeze(2) * self.std + self.mean).to(next(self.ae.parameters()).dtype)
        return rearrange(self.ae.decode(z).sample, "b c 1 h w -> b c h w")


class DepthEstimator:
    """Depth-Anything-V2-Large. Returns inverse depth in [0,1], near = 1."""

    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        mid = "depth-anything/Depth-Anything-V2-Large-hf"
        self.processor = AutoImageProcessor.from_pretrained(mid)
        self.model = (AutoModelForDepthEstimation
                      .from_pretrained(mid, torch_dtype=torch.float16)
                      .to(device).eval().requires_grad_(False))
        self.device = device

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        d = self.model(**inputs).predicted_depth[None].float()
        d = F.interpolate(d, size=(image.height, image.width),
                          mode="bilinear", align_corners=False)[0, 0]
        return (d - d.min()) / (d.max() - d.min() + 1e-6)


# ---------------------------------------------------------------- sampling

def pick_bucket(w, h):
    ar = math.log(w / h)
    return min(BUCKETS, key=lambda b: abs(math.log(b[0] / b[1]) - ar))


def resize_center_crop(img, tw, th):
    w, h = img.size
    s = max(tw / w, th / h)
    img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    w, h = img.size
    l, t = (w - tw) // 2, (h - th) // 2
    return img.crop((l, t, l + tw, t + th))


def prepare(img, txtlen, patch, txtmask):
    """Patchify a latent and build combined text+image position/mask tensors."""
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    ids = torch.zeros((h_, w_, 3), device=img.device)
    ids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    ids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    pos = repeat(ids, "h w c -> b (h w) c", b=b)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h p) (w q) -> b (h w) (c p q)", p=patch, q=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    return img, torch.cat((txtpos, pos), 1), torch.cat((txtmask, imgmask), 1)


def timesteps(seq_len, steps, mu=None):
    """Resolution-shifted flow schedule, t: 1 -> 0."""
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (MU_Y2 - MU_Y1) / (MU_X2 - MU_X1)
        mu = slope * seq_len + (MU_Y1 - slope * MU_X1)
    return (math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0))).tolist()


def forward_control(model, img, ctrl, context, t, pos, mask):
    """DiT forward with the depth latent concatenated on the channel dim."""
    x = model.first(torch.cat([img, ctrl], dim=-1))
    tv = model.tmlp(temb(t, model.config.tdim, device=x.device, dtype=x.dtype))
    tvec = model.tproj(tv)

    txtmask = _mask(mask[:, : context.shape[1]])
    context = model.txtmlp(model.txtfusion(context, mask=txtmask))

    txtlen, imglen = context.shape[1], x.shape[1]
    combined = torch.cat((context, x), dim=1)
    pad = (-combined.shape[1]) % 256
    if pad:
        combined = F.pad(combined, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad), value=False)
        pos = F.pad(pos, (0, 0, 0, pad))

    mask, freqs = _mask(mask), model.posemb(pos)
    for block in model.blocks:
        combined = block(combined, tvec, freqs, mask)
    return model.last(combined, tv)[:, txtlen: txtlen + imglen]


class DepthLoRAPipeline:
    def __init__(self, base_ckpt, lora_ckpt, rank=64, lora_scale=1.0, device="cuda"):
        self.device = device
        self.model = build_model(base_ckpt, lora_ckpt, rank, lora_scale, device)
        self.text = TextConditioner(device=device)
        self.vae = VAE(device=device)
        self.depth = DepthEstimator(device=device)

    @torch.no_grad()
    def __call__(self, image: Image.Image, prompt: str = "", steps: int = 8,
                 cfg: float = 0.0, mu: float | None = None, seed: int = 0):
        """Returns (output PIL, depth PIL). Turbo: steps=8 cfg=0 mu=1.15;
        Raw: steps=28-52 cfg=3.5 mu=None."""
        bw, bh = pick_bucket(*image.size)
        image = resize_center_crop(image.convert("RGB"), bw, bh)

        d = self.depth(image)
        depth_img = Image.fromarray((d.cpu().numpy() * 255).astype(np.uint8))
        depth_rgb = (d[None, None].repeat(1, 3, 1, 1).to(self.device) * 2 - 1)
        ctrl_lat = self.vae.encode(depth_rgb.to(torch.bfloat16))

        patch = self.model.config.patch
        noise = torch.randn(ctrl_lat.shape, device=self.device, dtype=torch.bfloat16,
                            generator=torch.Generator(self.device).manual_seed(seed))

        txt, tmask = self.text([prompt])
        x, pos, mask = prepare(noise, txt.shape[1], patch, tmask)
        ctrl, _, _ = prepare(ctrl_lat.to(torch.bfloat16), txt.shape[1], patch, tmask)
        if cfg > 0:
            untxt, unmask_t = self.text([""])
            _, unpos, unmask = prepare(noise, untxt.shape[1], patch, unmask_t)

        ts = timesteps(x.shape[1], steps, mu)
        img = x
        for tc, tp in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), tc, dtype=img.dtype, device=self.device)
            v = forward_control(self.model, img, ctrl, txt, t, pos, mask)
            if cfg > 0:
                un = forward_control(self.model, img, ctrl, untxt, t, unpos, unmask)
                v = v + cfg * (v - un)
            img = img + (tp - tc) * v

        h, w = ctrl_lat.shape[-2:]
        img = rearrange(img, "b (h w) (c p q) -> b c (h p) (w q)",
                        p=patch, q=patch, h=h // patch, w=w // patch)
        px = (self.vae.decode(img).clamp(-1, 1) * 0.5 + 0.5) * 255
        out = Image.fromarray(px[0].permute(1, 2, 0).float().cpu().byte().numpy())
        return out, depth_img
