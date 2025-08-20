import os
import json
import yaml
import torch
import random
import numpy as np
import torch.nn.functional as F
import clip
from data_utils import load_data
from visual_utils import patchify, viz_patches

# ─── Load Config ─────────────────────────────────────────────────────
cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

device = cfg.get("device", "cuda")
device = "cpu" if not torch.cuda.is_available() else device

num_samples = cfg["num_samples"]
dataset_name = cfg["dataset_name"]
model_id = cfg["model_id"]
keep_pct = cfg["keep_pct"]
viz = cfg["visualize"]

# ─── Load Model ──────────────────────────────────────────────────────
model, processor = clip.load(model_id, device)
model = model.float()

# ─── Model Input Preparation ─────────────────────────────────────────
def prepare_inputs(img, prompts):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        pixel_values = processor(img).unsqueeze(0).to(device)
        x = model.visual.conv1(pixel_values)
        B, D, H, W = x.shape
        tokens = x.view(B, D, -1).permute(0, 2, 1)

        cls_token = model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens += model.visual.positional_embedding.unsqueeze(0)

        tokens = model.visual.ln_pre(tokens).permute(1, 0, 2)
        tokens = model.visual.transformer(tokens).permute(1, 0, 2)

        cls = model.visual.ln_post(tokens[:, 0])
        if model.visual.proj is not None:
            cls = cls @ model.visual.proj
        cls /= cls.norm(dim=-1, keepdim=True)

        patch_tokens = tokens[:, 1:, :]

    return pixel_values, text_features, cls, patch_tokens

# ─── Patch Masking ───────────────────────────────────────────────────
def mask_patches(pixel_values, indices_to_remove):
    with torch.no_grad():
        x = model.visual.conv1(pixel_values)
        B, D, H, W = x.shape
        tokens = x.view(B, D, -1).permute(0, 2, 1)

        tokens[:, indices_to_remove, :] = 0  # zero out selected patches

        cls_token = model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens += model.visual.positional_embedding.unsqueeze(0)

        tokens = model.visual.ln_pre(tokens).permute(1, 0, 2)
        tokens = model.visual.transformer(tokens).permute(1, 0, 2)

        cls = model.visual.ln_post(tokens[:, 0])
        if model.visual.proj is not None:
            cls = cls @ model.visual.proj
        cls /= cls.norm(dim=-1, keepdim=True)

    return cls

# ─── Scoring Functions ───────────────────────────────────────────────
def get_confidence_score(pixel_values, indices_to_remove, text_features):
    masked_cls = mask_patches(pixel_values, indices_to_remove)
    logits = masked_cls @ text_features.T
    return logits.softmax(dim=-1).max(dim=-1)[0].item()

def get_feature_preservation_score(pixel_values, indices_to_remove, original_cls):
    masked_cls = mask_patches(pixel_values, indices_to_remove)
    return F.cosine_similarity(original_cls, masked_cls).item()

def fitness_function(pixel_values, indices, text_features, original_cls, weights=None):
    weights = weights or {'confidence': 0.4, 'feature': 0.6}
    conf = get_confidence_score(pixel_values, indices, text_features)
    feat = get_feature_preservation_score(pixel_values, indices, original_cls)
    return weights['confidence'] * conf + weights['feature'] * feat

# ─── Genetic Algorithm for Patch Selection ───────────────────────────
def genetic_algorithm(pixel_values, text_features, original_cls, num_patches, keep, population_size=20, generations=30, mutation_rate=0.1):
    population = [random.sample(range(num_patches), keep) for _ in range(population_size)]

    for _ in range(generations):
        scores = [fitness_function(pixel_values, ind, text_features, original_cls) for ind in population]
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]

        selected = np.random.choice(population_size, size=population_size, p=probs)
        parents = [population[i] for i in selected]

        next_gen = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % population_size]
            cp = random.randint(1, keep - 1)
            children = [p1[:cp] + p2[cp:], p2[:cp] + p1[cp:]]

            for child in children:
                if random.random() < mutation_rate:
                    child[random.randrange(keep)] = random.randrange(num_patches)
                child = sorted(set(child))[:keep]
                while len(child) < keep:
                    new_patch = random.randrange(num_patches)
                    if new_patch not in child:
                        child.append(new_patch)
                next_gen.append(child)

        population = next_gen

    return max(population, key=lambda ind: fitness_function(pixel_values, ind, text_features, original_cls))

# ─── Patch-Based CLIP Evaluation ─────────────────────────────────────
def patch_modified_clip(dataset, prompts, model, processor, device, keep_pct):
    results = []

    for idx, item in enumerate(dataset):
        print(f'{idx = }')
        img, label = item["img"], item["fine_label"]
        pixel_values, text_features, original_cls, patch_tokens = prepare_inputs(img, prompts)

        x = model.visual.conv1(processor(img).unsqueeze(0).to(device))
        B, D, N = x.shape[0], x.shape[1], x.numel() // (x.shape[0] * x.shape[1])
        x = x.view(B, D, -1).permute(0, 2, 1)

        num_patches = patch_tokens.size(1)
        keep = max(1, int(keep_pct * num_patches))
        iteration = 0

        while iteration < 10:
            selected_indices = genetic_algorithm(pixel_values, text_features, original_cls, num_patches, keep)

            cls = model.visual.class_embedding.unsqueeze(0).expand(1, -1, -1)
            sequence = torch.cat([cls, x], dim=1)
            pos = model.visual.positional_embedding[:sequence.size(1)].unsqueeze(0)
            sequence = model.visual.ln_pre(sequence + pos)

            keep_idx = torch.tensor([0] + [i + 1 for i in selected_indices], device=device)
            sequence = sequence[:, keep_idx, :]

            z = model.visual.transformer(sequence.permute(1, 0, 2))
            cls_token = model.visual.ln_post(z.permute(1, 0, 2)[:, 0])
            img_feat = cls_token @ model.visual.proj if model.visual.proj is not None else cls_token
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            probs = (100 * img_feat @ text_features.T).softmax(-1)
            pred = probs.argmax().item()

            print(f"Iteration {iteration}: Prediction={pred}, GT={label}")
            iteration += 1

            if pred == label:
                if viz:
                    patches = patchify(img, resolution=224, patch_size=16)
                    viz_patches(patches, topk=selected_indices, img_title=f"best_patches_{idx}_{pred = }_{label = }")
                break

        results.append({
            'image_id': item.get('id'),
            'selected_indices': selected_indices
        })

    return results

# ─── Main Function ───────────────────────────────────────────────────
def main():
    print(f"Evaluating on {'full' if num_samples == 0 else num_samples} samples of {dataset_name} dataset")
    dataset, prompts = load_data(dataset_name, num_samples)
    results = patch_modified_clip(dataset, prompts, model, processor, device, keep_pct)

    filename = f"{dataset_name}_{num_samples}_final_patches_{int(keep_pct * 100)}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {filename}")

# ─── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
