import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import transforms
except Exception:
    transforms = None

from dataset_patch_selection import PatchSelectionDataset
from model import PatchSelectionCNN


def build_dataloaders(
    jsonl_path: str,
    dataset_name: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    num_classes: int | None,
    val_ratio: float,
) -> Tuple[DataLoader, DataLoader, int]:
    t = None
    if transforms is not None:
        t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    full_ds = PatchSelectionDataset(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        split=split,
        num_classes=num_classes,
        img_size=img_size,
        transform=t,
    )

    n_total = len(full_ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, full_ds.num_classes


def train_one_epoch(model, loader, device, criterion, optimizer, scaler=None):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_f1 = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1

        # simple micro F1 over batch
        preds = (logits.sigmoid() > threshold).float()
        tp = (preds * y).sum().item()
        fp = (preds * (1 - y)).sum().item()
        fn = ((1 - preds) * y).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        total_f1 += f1
        total += 1

    return (total_loss / max(n_batches, 1)), (total_f1 / max(total, 1))


def main():
    parser = argparse.ArgumentParser(description="Train CNN to predict selected patch indices")
    parser.add_argument("--jsonl", type=str, default=os.path.join(os.path.dirname(__file__), "cifar100_10_final_patches_50.jsonl"))
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=196, help="Number of patch positions; if -1, infer from JSONL")
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "patch_cnn.pt"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    num_classes = None if args.num_classes == -1 else args.num_classes

    train_loader, val_loader, num_classes = build_dataloaders(
        jsonl_path=args.jsonl,
        dataset_name=args.dataset,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=num_classes,
        val_ratio=args.val_ratio,
    )

    model = PatchSelectionCNN(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_f1 = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_f1 {val_f1:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "num_classes": num_classes,
                "epoch": epoch,
            }, args.out)
            print(f"Saved best to {args.out}")


if __name__ == "__main__":
    main()
