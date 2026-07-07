"""Model surgery for Krea-2 depth control: expanded input layer + LoRA injection.

Imports SingleStreamDiT from the krea-2 repo (must be on sys.path).
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from mmdit import SingleMMDiTConfig, SingleStreamDiT

K2_RAW_CONFIG = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

LORA_TARGETS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
                "mlp.gate", "mlp.up", "mlp.down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        y = self.base(x)
        lora = (x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)
        return y + lora * self.scale


class ControlInputLayer(nn.Module):
    """Replaces model.first: takes (B, L, 2*C) = [noisy patches, control patches].

    Pretrained weights on the first half, zeros on the control half. Fully trained,
    fp32 master weights.
    """

    def __init__(self, pretrained: nn.Linear):
        super().__init__()
        in_f, out_f = pretrained.in_features, pretrained.out_features
        self.weight = nn.Parameter(torch.zeros(out_f, in_f * 2, dtype=torch.float32))
        self.bias = nn.Parameter(pretrained.bias.detach().float().clone())
        with torch.no_grad():
            self.weight[:, :in_f] = pretrained.weight.detach().float()

    def forward(self, x):
        return nn.functional.linear(
            x, self.weight.to(x.dtype), self.bias.to(x.dtype)
        )


def _get_module(root: nn.Module, path: str) -> nn.Module:
    mod = root
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def _set_module(root: nn.Module, path: str, new: nn.Module):
    parts = path.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], new)


def build_control_model(
    ckpt_path: str,
    rank: int = 64,
    alpha: float | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    targets: tuple[str, ...] = LORA_TARGETS,
) -> SingleStreamDiT:
    alpha = alpha if alpha is not None else rank
    with torch.device("meta"):
        model = SingleStreamDiT(K2_RAW_CONFIG)
    model.load_state_dict(load_file(ckpt_path), strict=True, assign=True)
    model = model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    model.first = ControlInputLayer(model.first).to(device)

    for i in range(K2_RAW_CONFIG.layers):
        for t in targets:
            path = f"blocks.{i}.{t}"
            base = _get_module(model, path)
            _set_module(model, path, LoRALinear(base, rank, alpha).to(device))
            _get_module(model, path).base.requires_grad_(False)

    return model


def trainable_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".A" in k or ".B" in k or k.startswith("first.")
    }


def load_trainable_state_dict(model: nn.Module, sd: dict[str, torch.Tensor]):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
