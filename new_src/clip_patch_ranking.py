import torch
import torch.nn.functional as F
import clip
from visual_utils import patchify, viz_patches, plot_heatmap_overlay, visualize_on_original
from data_utils import load_data
import yaml
import os
import random
import numpy as np
import json

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
keep_pct = cfg["keep_pct"]
viz = cfg['visualize']

# load model
model, processor = clip.load(model_id, device); model = model.float()

# Function to prepare model inputs
def prepare_inputs(img, prompts):
    toks = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(toks)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    pixel_values = processor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        x = model.visual.conv1(pixel_values)
        B, D, H, W = x.shape
        tokens = x.reshape(B, D, -1).permute(0, 2, 1)

        cls_token = model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens += model.visual.positional_embedding.unsqueeze(0)

        tokens = model.visual.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = model.visual.transformer(tokens)
        tokens = tokens.permute(1, 0, 2)

        orig_cls_token = tokens[:, 0, :]
        orig_cls_token = model.visual.ln_post(orig_cls_token)

        if model.visual.proj is not None:
            orig_cls_token = orig_cls_token @ model.visual.proj

        orig_cls_token /= orig_cls_token.norm(dim=-1, keepdim=True)
        patch_tokens = tokens[:, 1:, :]

    return pixel_values, text_features, orig_cls_token, patch_tokens

# Helper function to mask patches
def mask_patches(pixel_values, indices_to_remove):
    with torch.no_grad():
        x = model.visual.conv1(pixel_values)
        B, D, H, W = x.shape
        tokens = x.reshape(B, D, -1).permute(0, 2, 1)

        for idx in indices_to_remove:
            tokens[:, idx, :] = 0

        cls_token = model.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens += model.visual.positional_embedding.unsqueeze(0)

        tokens = model.visual.ln_pre(tokens)
        tokens = tokens.permute(1, 0, 2)
        tokens = model.visual.transformer(tokens)
        tokens = tokens.permute(1, 0, 2)

        masked_cls = tokens[:, 0, :]
        masked_cls = model.visual.ln_post(masked_cls)

        if model.visual.proj is not None:
            masked_cls = masked_cls @ model.visual.proj

        masked_cls /= masked_cls.norm(dim=-1, keepdim=True)

    return masked_cls

# Compute confidence score
def get_confidence_score(pixel_values, indices_to_remove, text_features):
    masked_cls_embedding = mask_patches(pixel_values, indices_to_remove)
    logits = masked_cls_embedding @ text_features.T
    probs = logits.softmax(dim=-1)
    return probs.max(dim=-1)[0].item()

# Compute feature preservation score
def get_feature_preservation_score(pixel_values, indices_to_remove, orig_cls_token):
    masked_cls_embedding = mask_patches(pixel_values, indices_to_remove)
    similarity = F.cosine_similarity(orig_cls_token, masked_cls_embedding).item()
    return similarity

# Fitness function combining multiple scores
def fitness_function(pixel_values, indices, text_features, orig_cls_token, weights=None):
    if weights is None:
        weights = {'confidence': 0.4, 'feature': 0.6}

    conf = get_confidence_score(pixel_values, indices, text_features)
    feat = get_feature_preservation_score(pixel_values,  indices, orig_cls_token)

    return weights['confidence'] * conf + weights['feature'] * feat

# Genetic Algorithm for patch selection
def genetic_algorithm(pixel_values, text_features, orig_cls_token, num_patches, keep, population_size=20, generations=30, mutation_rate=0.2):
    population = [random.sample(range(num_patches), keep) for _ in range(population_size)]

    for gen in range(generations):
        scores = [fitness_function(pixel_values, indiv, text_features, orig_cls_token) for indiv in population]
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]

        parent_idxs = np.random.choice(range(population_size), size=population_size, p=probs)
        parents = [population[i] for i in parent_idxs]

        next_gen = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % population_size]
            cp = random.randint(1, keep - 1)
            c1, c2 = p1[:cp] + p2[cp:], p2[:cp] + p1[cp:]
            for child in (c1, c2):
                if random.random() < mutation_rate:
                    idx = random.randrange(keep)
                    child[idx] = random.randrange(num_patches)
                child[:] = sorted(set(child))[:keep]
                while len(child) < keep:
                    new_patch = random.randrange(num_patches)
                    if new_patch not in child:
                        child.append(new_patch)
                next_gen.append(child)
        population = next_gen

    best = max(population, key=lambda indiv: fitness_function(pixel_values, indiv, text_features, orig_cls_token))
    return best


def patch_modified_clip(dataset, prompts, model, processor, device, keep_pct):
    toks = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(toks)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    results = []

    for idx, item in enumerate(dataset):
        print(f'{idx = }')
        img, label = item["img"], item["fine_label"]
        pixel_values, text_features, orig_cls_token, patch_tokens = prepare_inputs(img, prompts)
        img_input = processor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            x = model.visual.conv1(img_input)
        B,D,N = x.shape[0], x.shape[1], x.numel()//(x.shape[0]*x.shape[1])
        x = x.reshape(B, D, -1).permute(0,2,1)

        num_patches = patch_tokens.size(1)
        keep = max(1, int(keep_pct * num_patches))

        # select indices using Genetic Patch Prunning method
        iteration = 0
        while True:
            selected_indices = genetic_algorithm(pixel_values, text_features, orig_cls_token, num_patches, keep)

            cls = model.visual.class_embedding + torch.zeros(1, 1, D, device=device)
            seq_all = torch.cat([cls, x], dim=1)
            pos_all = model.visual.positional_embedding[:seq_all.size(1)].unsqueeze(0)
            seq_all = model.visual.ln_pre(seq_all + pos_all)
            keep_idx = torch.tensor([0] + [i + 1 for i in selected_indices], device=device)
            seq = seq_all[:, keep_idx, :]

            # Classification
            z = model.visual.transformer(seq.permute(1,0,2))
            z = model.visual.ln_post(z.permute(1,0,2)[:,0])

            img_f = (z @ model.visual.proj) if model.visual.proj is not None else z
            img_f /= img_f.norm(dim=-1,keepdim=True)

            sim2 = (100*img_f @ text_features.T).softmax(-1)
            pred = sim2.argmax().item()

            print(f"Iteration {iteration}: Prediction={pred}, GT={label}")
            iteration += 1

            if pred == label:
                if viz:
                    patches = patchify(img, resolution=224, patch_size=16)
                    viz_patches(patches, topk=selected_indices, img_title=f"best_patches_{idx}_{pred = }_{label = }")
                break

            if iteration >=10:
                break
            

            results.append({
                'image_id': item['id'] if 'id' in item else None,
                'selected_indices': selected_indices
            })


    return results


def main():

    if num_samples != 0:
        print(f"Evaluating on {num_samples} samples of {dataset_name} dataset")
    else:
        print(f"Evaluating on full {dataset_name} dataset")
    # load data & baseline    
    dataset, prompts = load_data(dataset_name, num_samples)
    # print(f'{prompts = }')
    results = patch_modified_clip(dataset, prompts, model, processor, device, keep_pct)

    # print(dataset)

    # for idx, item in enumerate(dataset):
    #     print(f'{item=}')
    #     img = item['img']
        
    #     pixel_values, text_features, orig_cls_token, patch_tokens = prepare_inputs(img, prompts)
        
    #     num_patches = patch_tokens.size(1)
    #     keep = int(keep_pct * num_patches)

    #     best_patches = genetic_algorithm(pixel_values, text_features, orig_cls_token, num_patches, keep)
    #     results.append({"image_id": idx, "patch_ids": best_patches})

    #     if viz:
    #         patches = patchify(img, resolution=224, patch_size=16)
    #         viz_patches(patches, topk=best_patches, img_title=f"best_patches_{idx}")

    filename = f"{dataset_name}_{num_samples}_final_patches_{int(keep_pct * 100)}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {filename}")



# Main execution
if __name__ == "__main__":

    main()
    # image_path = "path/to/your/image.jpg"
    # labels = ["label1", "label2", "ground_truth_label"]

    # pixel_values, text_features, orig_cls_token, patch_tokens = prepare_inputs(image_path, labels)
    # num_patches = patch_tokens.size(1)
    # keep = int(0.5 * num_patches)  # keeping 50% patches

    # best_patches = genetic_algorithm(pixel_values, text_features, orig_cls_token, num_patches, keep)

    # print(f"Best selected patch indices: {best_patches}")
