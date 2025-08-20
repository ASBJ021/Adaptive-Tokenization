import time
import torch
from datasets import load_dataset
import clip
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE = {DEVICE}")

NUM_SAMPLES = 1000
DATASET_NAME = "cifar100"
MODEL_ID = 'ViT-B/16'

# --- Dataset and Prompts ---
def load_data():
    dataset = load_dataset(DATASET_NAME, split="test")
    dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    classnames = dataset.features["fine_label"].names
    prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
    return dataset, prompts

# --- Original CLIP Inference ---
def original_clip(dataset, prompts, model_id=MODEL_ID):
    model, proc = clip.load(model_id, DEVICE)
    model = model.float()

    total_time, correct = 0.0, 0
    for item in dataset:
        image, label = item['img'], item['fine_label']
        image_input = proc(image).unsqueeze(0).to(DEVICE).float()
        text_inputs = clip.tokenize(prompts).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sim = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pred = sim[0].argmax().item()
        elapsed = time.time() - start

        total_time += elapsed
        correct += int(pred == label)

    return correct / len(dataset), total_time / len(dataset)

# --- Modified CLIP: Random Patch Removal Only ---
def modified_clip_dropout(dataset, prompts, model_id=MODEL_ID,
                          keep_pct=0.8, remove_stage='pre', seed=None):
    """
    Runs CLIP inference by randomly selecting a subset of visual patches.
    Uses all prompts for classification, ensuring proper accuracy computation.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model, proc = clip.load(model_id, DEVICE)
    model = model.float()

    # Precompute all text features once
    text_inputs_all = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features_all = model.encode_text(text_inputs_all)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

    total_time, correct = 0.0, 0
    for item in dataset:
        image, label = item['img'], item['fine_label']

        image_input = proc(image).unsqueeze(0).to(DEVICE).float()

        with torch.no_grad():
            start = time.time()
            # Extract patch tokens
            x = model.visual.conv1(image_input)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            B, N, D = x.shape
            keep_count = max(1, int(keep_pct * N))

            # Randomly select patch indices
            idx = torch.randperm(N, device=DEVICE)[:keep_count]
            idx, _ = torch.sort(idx)

            # Removal stage handling
            if remove_stage == 'pre':
                sel = x[:, idx, :]
                cls = model.visual.class_embedding + torch.zeros(B, 1, D, device=DEVICE)
                seq = torch.cat([cls, sel], dim=1)
                pos = model.visual.positional_embedding[: seq.size(1), :].unsqueeze(0)
                seq = model.visual.ln_pre(seq + pos)
            else:
                cls = model.visual.class_embedding + torch.zeros(B, 1, D, device=DEVICE)
                seq_full = torch.cat([cls, x], dim=1)
                pos_full = model.visual.positional_embedding[: seq_full.size(1), :].unsqueeze(0)
                seq_full = model.visual.ln_pre(seq_full + pos_full)
                keep_idx = torch.cat([torch.tensor([0], device=DEVICE), idx + 1])
                seq = seq_full[:, keep_idx, :]

            # Transformer & projection
            z = seq.permute(1, 0, 2)
            z = model.visual.transformer(z)
            z = z.permute(1, 0, 2)
            z = model.visual.ln_post(z[:, 0, :])
            img_feat = z @ model.visual.proj if model.visual.proj is not None else z

            # Compute similarity against all classes
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            sim2 = (100.0 * img_feat @ text_features_all.T).softmax(dim=-1)
            pred = sim2[0].argmax().item()
            elapsed = time.time() - start

        total_time += elapsed
        correct += int(pred == label)

    return correct / len(dataset), total_time / len(dataset)
# --- Main Execution ---
def main():
    dataset, prompts = load_data()
    orig_acc, orig_time = original_clip(dataset, prompts)

    keep_pcts = [i/10 for i in range(10, -1, -1)]
    stages = ['pre', 'post']
    records = []

    for stage in stages:
        for pct in keep_pcts:
            acc, t = modified_clip_dropout(
                dataset, prompts,
                keep_pct=pct,
                remove_stage=stage,
                seed=42
            )
            records.append({'pct': pct, 'stage': stage, 'acc': acc, 'avg_time': t})

    df = pd.DataFrame(records)

    if NUM_SAMPLES != 0:
        print(f"Evaluating on {NUM_SAMPLES} samples of {DATASET_NAME} dataset")
    else: 
        print(f"Evaluating on full {DATASET_NAME} dataset")

    # Print baseline
    print(f"Baseline (100% patches) - Accuracy: {orig_acc*100:.2f}%, Time: {orig_time:.4f}s")

    # Prepare wide-format results table
    acc_df = df.pivot(index='pct', columns='stage', values='acc')
    time_df = df.pivot(index='pct', columns='stage', values='avg_time')
    wide = pd.DataFrame({
        'pct': acc_df.index,
        'pre_acc': acc_df['pre'],
        'post_acc': acc_df['post'],
        'pre_avg_time': time_df['pre'],
        'post_avg_time': time_df['post'],
    }).reset_index(drop=True)

    # Format values
    wide['pct'] = wide['pct'].map(lambda x: f"{x:.1f}")
    wide['pre_acc'] = wide['pre_acc'].map(lambda x: f"{x:.2%}")
    wide['post_acc'] = wide['post_acc'].map(lambda x: f"{x:.2%}")
    wide['pre_avg_time'] = wide['pre_avg_time'].map(lambda x: f"{x:.4f}s")
    wide['post_avg_time'] = wide['post_avg_time'].map(lambda x: f"{x:.4f}s")

    # Print table header
    print("\npct  pre_acc  post_acc  pre_avg_time  post_avg_time")
    for _, row in wide.iterrows():
        print(f"{row['pct']:<4} {row['pre_acc']:>8} {row['post_acc']:>9} {row['pre_avg_time']:>13} {row['post_avg_time']:>14}")

if __name__ == "__main__":
    main()
