# Training a Krea-2 ControlNet-LoRA

Train a control adapter for [Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) on **any spatially-aligned control signal** — depth, canny edges, tile/blur, grayscale, or your own. The released depth LoRA was trained with exactly this code.

**How it works:** the control image is VAE-encoded with the same Qwen-Image VAE as the target image, and channel-concatenated to the noisy latent at every step (each DiT token: 64 → 128 dims). Only the expanded input projection + a LoRA (default rank 64) on all 28 blocks train — the 13B base stays frozen. Same recipe as BFL's Flux.1-Depth-dev-lora.

```
trainer/
├── prepare_data.py        # images + captions  →  latent shards (.npz)
├── train_control_lora.py  # latent shards      →  LoRA checkpoint
├── sampling.py            # unmodified sampler utils from the krea-2 repo
└── requirements.txt       # training extras (wandb, opencv, pyarrow)
```

```bash
pip install -r trainer/requirements.txt
```

## 1. Prepare data

You need **images + captions**. Control images are computed automatically for the built-in control types, or supplied by you (`custom`).

Built-in control types:

| `--control-type` | control signal | extractor |
|---|---|---|
| `depth` | inverse depth map (near = white) | Depth-Anything-V2-Large |
| `canny` | white edges on black | OpenCV Canny (`--canny-low/--canny-high`) |
| `tile` | blurred image | downscale ×`--tile-factor`, upscale back |
| `gray` | grayscale image | recolorization control |
| `custom` | anything | you provide paired control images |

### From a local folder

Images (`.jpg/.png/.webp/...`) in one directory. Captions come from either a `metadata.jsonl` (`{"file_name": "0001.jpg", "text": "a caption"}` per line) or per-image `.txt` sidecars (`0001.txt`); missing captions become empty strings (not recommended — caption quality matters).

```bash
python trainer/prepare_data.py --source folder \
    --input-dir ./my_images --out-dir ./data --control-type canny
```

For `custom`, put one control image per sample in a second folder, **same filename stem** as its source image (e.g. pose skeletons, segmentation maps, scribbles — rasterized to RGB):

```bash
python trainer/prepare_data.py --source folder \
    --input-dir ./my_images --control-dir ./my_pose_maps \
    --out-dir ./data --control-type custom
```

### From a HF parquet dataset

Streams shards without downloading the whole dataset — parallelize by giving each worker a `--shards` range:

```bash
export HF_TOKEN=...   # if the dataset is gated/private
python trainer/prepare_data.py --source hf \
    --dataset user/my-dataset --num-shards 158 --shards 0-19 \
    --out-dir /data --control-type depth \
    --image-col images --prompt-col prompts
```

### What it produces (the format the trainer reads)

Each sample is resized/center-cropped to the nearest ~1MP aspect bucket, then image and control are VAE-encoded:

```
data/
├── shard00000/
│   ├── <id>.npz          # latent  float16 (16, H/8, W/8)  — image latent
│   │                     # control float16 (16, H/8, W/8)  — control latent
│   │                     # prompt  str,  size (W, H)
│   ├── index.jsonl       # {"file": "shard00000/<id>.npz", "bucket": [W, H]} per line
│   └── _DONE             # marker: shard finished (safe to resume/re-run)
└── shard00001/ ...
```

If you build shards with your **own pipeline**, match this format and the trainer will consume it: the only hard requirements are that `control` is encoded with the Qwen-Image VAE (`Qwen/Qwen-Image`, subfolder `vae`, normalized with its `latents_mean/std`) and has the **same shape** as `latent`, i.e. the control is a pixel-aligned RGB rendering of your condition.

## 2. Train

```bash
hf download krea/Krea-2-Raw raw.safetensors --local-dir .

python trainer/train_control_lora.py \
    --data-dir ./data --ckpt-dir ./ckpts \
    --raw-ckpt ./raw.safetensors \
    --control-type canny
```

| flag | default | notes |
|---|---|---|
| `--control-type` | `depth` | label only (run name, wandb, ckpt metadata) — the signal comes from the data |
| `--rank` | 64 | LoRA rank; 862MB ckpt at r64 |
| `--lr` | 1e-4 | AdamW, linear warmup (`--warmup 200`) |
| `--batch-size` | 8 | per step, bucketed (same aspect ratio per batch) |
| `--grad-accum` | 4 | effective batch 32 |
| `--max-steps` | 6000 | the depth LoRA converged well by 6k |
| `--caption-dropout` | 0.1 | **keep this on** — without it the CFG/unconditional pathway degrades |
| `--save-every` | 500 | writes `ckpts/<run>/step_XXXXXX.safetensors` |
| `--resume` | — | path to a saved checkpoint; resumes step from its metadata |
| `--wandb-project` | `krea2-controlnet` | logging is on iff `WANDB_API_KEY` is set and not `--no-wandb` |
| `--bucket-uri` | off | optional `hf://buckets/...` checkpoint upload |
| `--max-samples` | 0 | debug: cap dataset size for smoke tests |

Hardware: the 13B DiT trains in bf16 with gradient checkpointing plus Qwen3-VL-4B (text conditioning) and the VAE resident — an 80 GB GPU (A100/H100 class) handles batch 8 at ~1MP. Checkpoints contain only the trainable weights (LoRA A/B + expanded input layer), so the base checkpoint is always needed at inference.

Validation images (logged to wandb every `--val-every` steps) generate from the first training sample's control — early steps look like the base model ignoring control; structure should lock in as the zero-initialized control weights warm up.

## 3. Use the checkpoint

The saved file is directly compatible with the inference code in the repo root:

```bash
python inference.py photo.jpg -p "..." --lora ckpts/<run>/step_006000.safetensors
```

One caveat: `pipeline.py` computes the control with a **depth extractor** at inference. For any other control type, swap `DepthEstimator` in `pipeline.py` for your extractor (the same transform you used in `prepare_data.py` — e.g. Canny), so inference-time control images match what the LoRA saw in training. Everything downstream (VAE-encode → channel-concat) is control-agnostic.

Although trained on Raw, checkpoints also work on **Krea-2-Turbo** (8 steps, cfg 0) via `--base turbo`.

## Adding a new control type

1. In `prepare_data.py`, write an extractor class: `__call__(images: list[PIL]) -> list[(3, h, w) tensor in [-1, 1]]`, each output the same size as its input. Register it in `build_extractor()` and add the name to the `--control-type` choices.
2. Prep data and train with your new type — no trainer changes needed.
3. For inference, mirror the extractor in `pipeline.py` (replace `DepthEstimator`).

Or skip step 1 entirely: precompute control images offline and use `--control-type custom --control-dir ...`.

**What makes a good control signal here:** this is channel-concat control, so the condition must be a dense, pixel-aligned image (depth, edges, segmentation, blur, scribbles all work). Sparse/abstract conditions (keypoints, text layout) need to be rasterized to an image first.
