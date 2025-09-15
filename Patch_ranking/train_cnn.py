import os
import json
import csv
import argparse
from typing import Optional
import yaml
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import local modules in both module and script contexts

from dataset import PatchIndexDataset, _load_jsonl, split_dataset
from cnn import SmallCNN

# Import dataset helper from new_src with a safe fallback

from new_src.data_utils import load_data_normal




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN to predict selected patch indices")
    p.add_argument("--dataset_name", type=str, default="cifar100", help="HF dataset name (e.g., cifar100)")
    p.add_argument("--annotations", type=str, default="Patch_ranking/cifar100_10_final_patches_50.jsonl",
                   help="Path to JSONL annotations with image_id and selected_indices")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="Patch_ranking/checkpoints")
    p.add_argument("--val_split", type=float, default=0.15, help="Unused; kept for compatibility")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", type=str, default=os.path.join("Patch_ranking", "config.yaml"),
                   help="Path to YAML config; if exists, values override CLI")
    p.add_argument("--predictions_jsonl", type=str, default=os.path.join("Patch_ranking", "checkpoints", "predictions.jsonl"),
                   help="Write test predictions with predicted_indices to this JSONL path")
    p.add_argument("--pred_thresh", type=float, default=0.5, help="Sigmoid threshold for predicting indices")
    return p.parse_args()


def accuracy_at_threshold(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5) -> float:
    """Simple multi-label accuracy: fraction of correctly predicted bits over all bits."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= thresh).float()
        correct = (preds == targets).float().mean().item()
    return correct


def _resolve_path(path: str, repo_root: str) -> str:
    """Resolve a possibly relative path against common roots.
    Tries the path as-is, then relative to repo_root, and also normalizes
    the leading segment case for 'patch_ranking'/'Patch_ranking'.
    Returns an absolute path if a candidate exists; otherwise returns the
    path joined to repo_root for consistency.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path

    candidates = []
    candidates.append(os.path.abspath(path))
    candidates.append(os.path.abspath(os.path.join(repo_root, path)))

    # Normalize first path segment case
    parts = path.replace("\\", "/").split("/")
    if parts:
        first = parts[0]
        rest = parts[1:]
        if first.lower() == "patch_ranking":
            alt1 = os.path.join("patch_ranking", *rest)
            alt2 = os.path.join("Patch_ranking", *rest)
            candidates.append(os.path.abspath(alt1))
            candidates.append(os.path.abspath(os.path.join(repo_root, alt1)))
            candidates.append(os.path.abspath(alt2))
            candidates.append(os.path.abspath(os.path.join(repo_root, alt2)))

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Fallback to repo_root join
    return os.path.abspath(os.path.join(repo_root, path))


def _load_and_apply_config(args: argparse.Namespace, repo_root: str) -> argparse.Namespace:
    """Load YAML config (if present) and override matching argparse fields.
    Resolves the config path relative to repo_root when needed.
    """
    cfg_path = args.config
    if cfg_path and not os.path.isabs(cfg_path):
        # Try as-is, then repo_root/config
        if not os.path.isfile(cfg_path):
            alt = os.path.join(repo_root, cfg_path)
            if os.path.isfile(alt):
                cfg_path = alt
    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


def main() -> None:
    args = parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))

    # Load YAML config if present and override args
    args = _load_and_apply_config(args, repo_root)

    # Resolve key filesystem paths relative to repo root
    args.annotations = _resolve_path(args.annotations, repo_root)
    args.save_dir = _resolve_path(args.save_dir, repo_root)
    args.predictions_jsonl = _resolve_path(args.predictions_jsonl, repo_root)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f'{args = }')

    # Seeding for reproducibility (affects split and dataloader shuffling)
   
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if np is not None:
        np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load annotations to determine num_samples and num_classes
    records = _load_jsonl(args.annotations)
    if not records:
        raise RuntimeError(f"No records found in {args.annotations}")

    # Load full HF dataset; we will filter by matching image_id via enumerate in PatchIndexDataset
    # We use NUM_SAMPLES as the dataset size upper bound; pick a safe ceiling from annotations
    num_samples = max(r.get("image_id", 0) for r in records) + 1
    ds, _ = load_data_normal(args.dataset_name, num_samples)

    # Build filtered dataset (only images whose index matches JSONL image_id)
    full_dataset = PatchIndexDataset(ds=ds, jsonl_path=args.annotations, img_size=args.img_size)
    num_classes = full_dataset.num_classes

    # Split into 70/15/15
    train_subset, val_subset, test_subset = split_dataset(full_dataset, ratios=(0.7, 0.15, 0.15), seed=args.seed)


    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # print(f'{train_loader = }')

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Model, optim, loss
    model = SmallCNN(num_outputs=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss: Optional[float] = None
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    history_csv = os.path.join(args.save_dir, "history.csv")
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for imgs, targets in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy_at_threshold(logits.detach(), targets)
            steps += 1

        avg_loss = total_loss / max(1, steps)
        avg_acc = total_acc / max(1, steps)

        log = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_bit_acc@0.5": avg_acc,
        }

        # Validation
        model.eval()
        v_total = 0.0
        v_acc = 0.0
        v_steps = 0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Val   {epoch}/{args.epochs}", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(imgs)
                loss = criterion(logits, targets)
                v_total += loss.item()
                v_acc += accuracy_at_threshold(logits, targets)
                v_steps += 1

        val_loss = v_total / max(1, v_steps)
        val_acc = v_acc / max(1, v_steps)
        log.update({
            "val_loss": val_loss,
            "val_bit_acc@0.5": val_acc,
        })

        # Save best
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, f"best_epoch_{epoch}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "args": vars(args),
            }, ckpt_path)

        # Persist per-epoch metrics
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        print(json.dumps(log))
        history_rows.append(log)

    # Final save
    final_path = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "args": vars(args),
    }, final_path)
    print(f"Saved final model to {final_path}")

    # Write history.csv for easy plotting of curves
    if history_rows:
        fieldnames = list(history_rows[0].keys())
        with open(history_csv, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in history_rows:
                writer.writerow(r)

    # Evaluate on test split
    model.eval()
    t_total = 0.0
    t_acc = 0.0
    t_steps = 0
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Test", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, targets)
            t_total += loss.item()
            t_acc += accuracy_at_threshold(logits, targets)
            t_steps += 1
    test_loss = t_total / max(1, t_steps)
    test_acc = t_acc / max(1, t_steps)
    print(json.dumps({"test_loss": test_loss, "test_bit_acc@0.5": test_acc}))

    # Export predicted_indices for the test split
    pred_ds = PatchIndexDataset(
        ds=full_dataset.ds,
        jsonl_path=None,
        num_classes=full_dataset.num_classes,
        img_size=args.img_size,
        transform=full_dataset.transform,
        items=test_subset.items,
        return_index=True,
    )
    pred_loader = DataLoader(
        pred_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    os.makedirs(os.path.dirname(args.predictions_jsonl), exist_ok=True)
    with torch.no_grad():
        with open(args.predictions_jsonl, "w", encoding="utf-8") as pf:
            for imgs, targets, ds_indices in tqdm(pred_loader, desc="Predict", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu()
                targets_cpu = targets.cpu()
                for b in range(probs.size(0)):
                    p = probs[b]
                    pred_idx = (p >= args.pred_thresh).nonzero(as_tuple=False).view(-1).tolist()
                    gt_idx = (targets_cpu[b] >= 0.5).nonzero(as_tuple=False).view(-1).tolist()
                    record = {
                        "image_id": int(ds_indices[b].item()),
                        "predicted_indices": pred_idx,
                        "selected_indices": gt_idx,
                    }
                    pf.write(json.dumps(record) + "\n")
    print(f"Wrote predictions to {args.predictions_jsonl}")


if __name__ == "__main__":
    main()
