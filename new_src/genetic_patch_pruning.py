import os
import json
import torch

from data_utils import load_data_normal
from visual_utils import patchify, viz_patches

from genetic_algo.config import load_config, resolve_device, default_config_path
from genetic_algo.clip_model import load_clip
from genetic_algo.runner import patch_modified_clip
import time


# ─── Load Config ─────────────────────────────────────────────────────
cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
print(f'{cfg_path = }')
cfg = load_config(cfg_path)
print(f'{cfg = }')
device = resolve_device(cfg)

num_samples = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
data_split = cfg['split']
model_id = cfg["model_id"]
keep_pct = cfg["keep_pct"]
viz = cfg["visualize"]
optimize_keep = cfg.get("optimize_keep", False)
min_keep_pct = cfg.get("min_keep_pct", 0.1)
max_keep_pct = cfg.get("max_keep_pct", 0.9)
keep_penalty = cfg.get("keep_penalty", 0.1)

# ─── Load Model ──────────────────────────────────────────────────────
model, processor = load_clip(model_id, device)


# ─── Main Function ───────────────────────────────────────────────────
def main():

    print(
        f"Evaluating on {'full' if num_samples == 0 else num_samples} samples of {dataset_name} dataset"
    )

    dataset, prompts = load_data_normal(dataset_name, num_samples, SPLIT=data_split)
    # print(f'{prompts =}')

    out_path_jsonl = f"{dataset_name}_{num_samples}_final_patches_{int(keep_pct * 100)}.jsonl"
    # Preserve original behavior: override path with hard-coded target
    out_path_jsonl = (
        f"/home/utn/firi22ka/Desktop/jenga/Adaptive-Tokenization/new_src/{dataset_name}_{num_samples}_{time.time()}.jsonl"
    )

    _ = patch_modified_clip(
        dataset,
        prompts,
        model,
        processor,
        device,
        keep_pct,
        out_path_jsonl,
        viz=viz,
        patchify_fn=patchify,
        viz_patches_fn=viz_patches,
        optimize_keep=optimize_keep,
        min_keep_pct=min_keep_pct,
        max_keep_pct=max_keep_pct,
        keep_penalty=keep_penalty,
    )

    print(
        f"Per-image results saved line-by-line to {out_path_jsonl} (resume supported)."
    )


# ─── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
