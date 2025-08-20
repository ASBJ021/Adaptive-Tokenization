import time
import math
import torch
from torch.nn.functional import normalize, cosine_similarity
from datasets import load_dataset
import clip
import pandas as pd
from tinyclip_bigclip_pipeline import select_top_k_indices, get_patch_tokens, get_text_embedding
from visualize import visualize_on_original, viz_patches, patchify, plot_heatmap_overlay
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE = {DEVICE}")

NUM_SAMPLES = 3
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

    accuracy = correct / len(dataset)
    avg_time = total_time / len(dataset)
    return accuracy, avg_time

# --- Modified CLIP with Patch Selection Strategies ---

def modified_clip_dropout(dataset, prompts, model_id=MODEL_ID,
                          keep_pct=0.8, strategy='random', seed=None):
    """
    Runs CLIP inference while selecting a subset of visual patches by strategy.

    Args:
        dataset: HuggingFace dataset of images/labels.
        prompts: List of text prompts for class labels.
        model_id: CLIP model identifier.
        keep_pct: Fraction of patches to keep (0.0 <= keep_pct <= 1.0).
        strategy: One of ['random', 'uniform', 'similarity'].
        seed: Optional seed for reproducibility.
    Returns:
        Tuple of (accuracy, avg_inference_time).
    """
    if seed is not None:
        torch.manual_seed(seed)

    model, proc = clip.load(model_id, DEVICE)
    model = model.float()

    patch_size=16
    grid_cols = grid_rows = 224 // patch_size

    # Precompute all text features once
    text_inputs_all = clip.tokenize(prompts).to(DEVICE)
    with torch.no_grad():
        text_features_all = model.encode_text(text_inputs_all)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)
    # print(f'{text_features_all.shape = }')

    total_time, correct = 0.0, 0
    for item in dataset:
        image, label = item['img'], item['fine_label']
        prompt_feat = text_features_all[label:label+1]
        # print(f'{prompt_feat.shape = }')
        image.show()
        image_input = proc(image).unsqueeze(0).to(DEVICE).float()

        with torch.no_grad():
            start = time.time()
            # Extract patch tokens
            x = model.visual.conv1(image_input)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            B, N, D = x.shape
            keep_count = max(1, int(keep_pct * N))

            # Choose indices to keep
            if strategy == 'random':
                idx = torch.randperm(N, device=DEVICE)[:keep_count]
            elif strategy == 'uniform':
                grid = int(math.sqrt(N))
                step = max(1, grid // int(math.sqrt(keep_count)))
                coords = [(i, j) for i in range(0, grid, step) for j in range(0, grid, step)]
                idx = torch.tensor([i*grid + j for i, j in coords], device=DEVICE)[:keep_count]

            elif strategy == "similarity":


                # Compute patch-level features for similarity
                pos = model.visual.positional_embedding[1:N+1].unsqueeze(0)  # [1, N, D]
                x_pe = model.visual.ln_pre(x + pos)  # [B, N, D]
                if model.visual.proj is not None:
                    # Project into embedding space
                    patch_embed = x_pe @ model.visual.proj  # [B, N, dim]
                else:
                    patch_embed = x_pe
                    
                patch_embed = patch_embed.squeeze(0)  # [N, dim]
                # Normalize embeddings
                patch_norm = normalize(patch_embed, dim=-1)
                prompt_norm = normalize(prompt_feat.squeeze(0), dim=-1)
                # Cosine similarity
                sims = patch_norm @ prompt_norm  # [N]
                # sims = (100.0 * patch_norm @ prompt_norm.T).softmax(dim=-1)
                # sims = cosine_similarity(patch_norm, prompt_norm, dim=-1)
                # Select top-k patches
                k = max(1, int(keep_pct * N))
                # idx = sims.topk(k).indices
                idx = torch.topk(sims, k).indices
                # print(f'{idx = }')

                plot_heatmap_overlay(image, idx,sims.cpu().numpy(), (grid_rows, grid_cols), alpha=0.4)



            else:
                raise ValueError(f"Unknown strategy {strategy}")
            
            idx, _ = torch.sort(idx)
            # print(f'sorted {idx = }')
            img_name = f'{strategy}_{keep_pct}_{label}'

            
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
        
        
        # visualize(image, idx, model, proc, img_name)
        # 4) SHOW patch grid with highlights
        
        patches = patchify(image, resolution=224, patch_size=16)
        viz_patches(patches, topk=idx.cpu(), img_title=img_name)
 

        total_time += elapsed
        correct += int(pred == label)

    return correct / len(dataset), total_time / len(dataset)



# --- Main: Evaluate Baseline and Strategies ---
def main():
    dataset, prompts = load_data()
    orig_acc, orig_time = original_clip(dataset, prompts)

    strategies = ['random', 'uniform', 'similarity']
    keep_pcts = [i/10 for i in range(10, 7, -1)]  # 1.0 down to 0.0 range(start, stop, step)
    # print(keep_pcts)

    # assert False

    records = []
    for strat in strategies:
        for pct in keep_pcts:
            acc, t = modified_clip_dropout(dataset, prompts,
                                   keep_pct=pct,
                                   strategy=strat,
                                   seed=42)
            records.append({
                'keep_pct': pct,
                f'{strat}_acc': acc,
                f'{strat}_avg_time': t
            })

    

    # if NUM_SAMPLES != 0:
    #     print(f"Evaluating on {NUM_SAMPLES} samples of {DATASET_NAME} dataset")
    # else: 
    #     print(f"Evaluating on full {DATASET_NAME} dataset")



    # # Print baseline
    # print(f"Baseline (100% patches) - Accuracy: {orig_acc*100:.2f}%, Time: {orig_time:.4f}s")

    # df = pd.DataFrame(records)

    # # Format tables using DataFrame.map
    # acc_df = df.pivot(index='keep_pct', columns='strategy', values='accuracy')
    # time_df = df.pivot(index='keep_pct', columns='strategy', values='avg_time')
    # acc_table = acc_df.map(lambda x: f"{x:.2%}")
    # time_table = time_df.map(lambda x: f"{x:.4f}s")

    # # Print tables
    # print("\nAccuracy table:")
    # print(acc_table.to_string())
    # print("\nAverage inference time table:")
    # print(time_table.to_string())


    # combine results into one table
    # df = pd.DataFrame(records)
    # # pivot so that each strategy's metrics align with keep_pct
    # wide = df.groupby('keep_pct').agg({
    #     f'{strat}_acc': 'first' for strat in strategies
    # })
    # for strat in strategies:
    #     wide[f'{strat}_avg_time'] = df.groupby('keep_pct')[f'{strat}_avg_time'].first()
    # wide = wide.reset_index()

    # # format numeric columns
    # for strat in strategies:
    #     wide[f'{strat}_acc'] = wide[f'{strat}_acc'].map(lambda x: f"{x*100:.2f}%")
    #     wide[f'{strat}_avg_time'] = wide[f'{strat}_avg_time'].map(lambda x: f"{x:.4f}s")

    # print("Results:")
    # print(wide.to_string(index=False))

if __name__ == "__main__":
    main()
