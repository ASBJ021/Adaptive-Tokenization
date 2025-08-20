# main.py

import os
import yaml
import torch
import pandas as pd

from data_utils import load_data
from model_utils import original_clip, modified_clip_dropout
from compare import display_comparison_tables

def main():
    # ─── load config.yaml ────────────────────────────────────────────────
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        device = "cpu"

    num_samples  = cfg["num_samples"]
    dataset_name = cfg["dataset_name"]
    model_id     = cfg["model_id"]
    vis = cfg["visualize"]
    # ───────────────────────────────────────────────────────────────────────

    # 1) sampling info
    if num_samples != 0:
        print(f"Evaluating on {num_samples} samples of {dataset_name} dataset")
    else:
        print(f"Evaluating on full {dataset_name} dataset")

    # 2) load data & baseline
    dataset, prompts = load_data(dataset_name, num_samples)
    orig_acc, orig_time = original_clip(dataset, prompts, model_id, device)

    # 3) print baseline
    print(f"\nBaseline (100% patches) - Accuracy: {orig_acc*100:.2f}%, Time: {orig_time:.4f}s")

    # 4) run modified versions
    strategies = ['random', 'uniform', 'similarity']
    keep_pcts  = [0.9, 0.8]  # adjust as desired

    records = []
    for strat in strategies:
        for pct in keep_pcts:
            acc, avg_time = modified_clip_dropout(
                dataset, prompts,
                model_id,
                device,
                keep_pct=pct,
                strategy=strat,
                seed=42,
                visualize=vis
            )
            # record both generic and strat‐specific fields
            records.append({
                'strategy':      strat,
                'keep_pct':      pct,
                'accuracy':      acc,
                'avg_time':      avg_time,
                f'{strat}_acc':       acc,
                f'{strat}_avg_time':  avg_time,
            })

    display_comparison_tables(records, strategies)
            

    

if __name__ == "__main__":
    main()
